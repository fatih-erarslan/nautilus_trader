"""
Yahoo Finance news source implementation for stock market news
"""
import re
import aiohttp
import logging
from typing import List, AsyncIterator, Dict, Any, Optional
from datetime import datetime
from . import NewsSource, NewsSourceError
from ..models import NewsItem


logger = logging.getLogger(__name__)


class YahooFinanceSource(NewsSource):
    """Yahoo Finance news source for market news and analysis"""
    
    def __init__(self):
        """Initialize Yahoo Finance news source (no API key required)"""
        super().__init__("yahoo_finance")
        self.base_url = "https://query1.finance.yahoo.com/v1"
    
    async def fetch_latest(self, limit: int = 100) -> List[NewsItem]:
        """
        Fetch latest news articles from Yahoo Finance
        
        Args:
            limit: Maximum number of articles to fetch
            
        Returns:
            List of NewsItem objects
        """
        url = f"{self.base_url}/finance/news"
        params = {"count": limit}
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        raise NewsSourceError(f"Yahoo Finance API error: {response.status}")
                    
                    data = await response.json()
                    
                    return [
                        self._parse_article(item)
                        for item in data.get("items", [])
                        if item.get("type") == "STORY"
                    ]
                    
            except aiohttp.ClientError as e:
                raise NewsSourceError(f"Yahoo Finance connection error: {str(e)}") from e
    
    def _parse_article(self, item: Dict[str, Any]) -> NewsItem:
        """Parse Yahoo Finance article data into NewsItem"""
        # Extract entities from the article
        entities = []
        for entity in item.get("entities", []):
            if "term" in entity:
                entities.append(entity["term"])
        
        # Convert Unix timestamp to datetime
        timestamp = datetime.fromtimestamp(item.get("published_at", 0))
        
        # Detect earnings news
        earnings_info = self._detect_earnings_news(item["title"])
        
        metadata = {
            "publisher": item.get("publisher", "Yahoo Finance"),
            "type": item.get("type"),
            **earnings_info
        }
        
        return NewsItem(
            id=f"yahoo-{item['uuid']}",
            title=item['title'],
            content=item.get('summary', ''),
            source=self.source_name,
            timestamp=timestamp,
            url=item['link'],
            entities=entities,
            metadata=metadata
        )
    
    def _detect_earnings_news(self, title: str) -> Dict[str, Any]:
        """
        Detect if article is earnings-related and extract metrics
        
        Args:
            title: Article title
            
        Returns:
            Dictionary with earnings information
        """
        result = {"is_earnings": False}
        
        # Earnings beat patterns with percentage
        beat_patterns = [
            r'beats?\s+.*?(?:earnings?|EPS|revenue).*?by\s+(\d+)%?',
            r'(?:beats?|exceed\w*|surpass\w*)\s+(?:earnings?|EPS|revenue|estimates?)(?:\s+by)?\s+(\d+)%?',
            r'(\d+)%?\s+(?:earnings?|EPS|revenue)\s+(?:beat|exceed|surpass)',
            r'(?:earnings?|EPS|revenue)\s+.*?(\d+)%?\s+(?:above|over|better)'
        ]
        
        # Check for earnings beat with percentage
        for pattern in beat_patterns:
            beat_match = re.search(pattern, title, re.IGNORECASE)
            if beat_match:
                result["is_earnings"] = True
                result["earnings_beat"] = True
                try:
                    surprise_pct = float(beat_match.group(1)) / 100
                    result["earnings_surprise"] = surprise_pct
                except:
                    result["earnings_surprise"] = 0.10  # Default 10% beat
                return result
        
        # Earnings miss patterns
        miss_pattern = r'(?:miss\w*|fall\w* short|below)\s+(?:earnings?|EPS|revenue|estimates?)'
        
        # Check for earnings miss
        if re.search(miss_pattern, title, re.IGNORECASE):
            result["is_earnings"] = True
            result["earnings_beat"] = False
            result["earnings_surprise"] = -0.05  # Default 5% miss
        
        # Check for general earnings mention
        elif re.search(r'\b(?:earnings?|EPS|revenue|Q[1-4]|report\w*)\b', title, re.IGNORECASE):
            result["is_earnings"] = True
        
        return result
    
    async def fetch_analyst_ratings(self, limit: int = 20) -> List[NewsItem]:
        """
        Fetch analyst ratings and recommendations
        
        Args:
            limit: Maximum number of ratings to fetch
            
        Returns:
            List of NewsItem objects with analyst ratings
        """
        url = f"{self.base_url}/finance/news"
        params = {
            "count": limit,
            "category": "analyst-ratings"
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        raise NewsSourceError(f"Yahoo Finance API error: {response.status}")
                    
                    data = await response.json()
                    
                    items = []
                    for item in data.get("items", []):
                        news_item = self._parse_article(item)
                        
                        # Extract analyst rating information
                        rating_info = self._extract_analyst_rating(news_item.title)
                        news_item.metadata.update(rating_info)
                        
                        items.append(news_item)
                    
                    return items
                    
            except aiohttp.ClientError as e:
                raise NewsSourceError(f"Yahoo Finance connection error: {str(e)}") from e
    
    def _extract_analyst_rating(self, title: str) -> Dict[str, Any]:
        """Extract analyst rating information from title"""
        info = {}
        
        # Extract analyst firm
        analyst_pattern = r'^([A-Za-z\s&]+?)(?:\s+(?:upgrade|downgrade|initiate|maintain|raise|lower))'
        analyst_match = re.match(analyst_pattern, title, re.IGNORECASE)
        if analyst_match:
            info["analyst"] = analyst_match.group(1).strip()
        
        # Detect rating changes
        if re.search(r'\bupgrade', title, re.IGNORECASE):
            info["rating_change"] = "upgrade"
            
            # Extract new rating
            rating_pattern = r'to\s+(Buy|Hold|Sell|Outperform|Underperform|Neutral)'
            rating_match = re.search(rating_pattern, title, re.IGNORECASE)
            if rating_match:
                info["new_rating"] = rating_match.group(1).title()
        
        elif re.search(r'\bdowngrade', title, re.IGNORECASE):
            info["rating_change"] = "downgrade"
        
        # Extract price target
        pt_pattern = r'(?:PT|price target|target).*?\$?(\d+(?:\.\d+)?)'
        pt_match = re.search(pt_pattern, title, re.IGNORECASE)
        if pt_match:
            info["price_target"] = float(pt_match.group(1))
        
        return info
    
    async def get_trending_tickers(self) -> List[str]:
        """
        Get currently trending stock tickers
        
        Returns:
            List of trending ticker symbols
        """
        url = f"{self.base_url}/finance/trending/US"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise NewsSourceError(f"Yahoo Finance API error: {response.status}")
                    
                    data = await response.json()
                    
                    tickers = []
                    results = data.get("finance", {}).get("result", [])
                    
                    if results:
                        quotes = results[0].get("quotes", [])
                        tickers = [quote["symbol"] for quote in quotes if "symbol" in quote]
                    
                    return tickers
                    
            except aiohttp.ClientError as e:
                raise NewsSourceError(f"Yahoo Finance connection error: {str(e)}") from e
    
    async def stream(self) -> AsyncIterator[NewsItem]:
        """
        Stream real-time news from Yahoo Finance
        Note: Yahoo Finance doesn't provide WebSocket streaming, so this polls periodically
        
        Yields:
            NewsItem objects as they arrive
        """
        import asyncio
        
        seen_ids = set()
        poll_interval = 30  # seconds
        
        while True:
            try:
                # Fetch latest news
                items = await self.fetch_latest(limit=20)
                
                # Yield only new items
                for item in items:
                    if item.id not in seen_ids:
                        seen_ids.add(item.id)
                        yield item
                
                # Keep seen_ids from growing too large
                if len(seen_ids) > 1000:
                    seen_ids = set(list(seen_ids)[-500:])
                
                await asyncio.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Stream error: {str(e)}")
                await asyncio.sleep(poll_interval * 2)  # Back off on error