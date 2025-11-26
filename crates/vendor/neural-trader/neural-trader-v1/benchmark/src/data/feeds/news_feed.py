"""
News feed handler for real-time financial news integration
Aggregates news from multiple sources for market impact analysis
"""
import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Set, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import re

from ..realtime_manager import DataFeed, DataPoint

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """Individual news item"""
    title: str
    summary: str
    url: str
    source: str
    published_at: datetime
    symbols: List[str]
    sentiment: Optional[float] = None  # -1 to 1
    impact_score: Optional[float] = None  # 0 to 1
    category: Optional[str] = None
    language: str = "en"
    metadata: Dict[str, Any] = None


class NewsFeed(DataFeed):
    """News feed handler for financial news"""
    
    # News sources configuration
    SOURCES = {
        'newsapi': {
            'url': 'https://newsapi.org/v2/everything',
            'requires_key': True,
            'rate_limit': 1000  # per day
        },
        'finnhub_news': {
            'url': 'https://finnhub.io/api/v1/news',
            'requires_key': True,
            'rate_limit': 60  # per minute
        },
        'alpha_vantage_news': {
            'url': 'https://www.alphavantage.co/query',
            'requires_key': True,
            'rate_limit': 5  # per minute
        }
    }
    
    # Financial keywords for filtering
    FINANCIAL_KEYWORDS = [
        'earnings', 'revenue', 'profit', 'loss', 'dividend', 'merger', 
        'acquisition', 'ipo', 'stock', 'shares', 'market', 'trading',
        'financial', 'quarter', 'guidance', 'forecast', 'outlook',
        'sec filing', 'regulations', 'federal reserve', 'interest rate',
        'inflation', 'gdp', 'unemployment', 'economic', 'fiscal'
    ]
    
    # Sentiment keywords
    POSITIVE_KEYWORDS = [
        'growth', 'increase', 'rise', 'surge', 'boom', 'profit', 'gain',
        'success', 'bullish', 'upgrade', 'beat', 'exceeded', 'strong'
    ]
    
    NEGATIVE_KEYWORDS = [
        'decline', 'fall', 'drop', 'crash', 'loss', 'cut', 'reduce',
        'bearish', 'downgrade', 'miss', 'weak', 'concern', 'risk'
    ]
    
    def __init__(self, symbols: List[str] = None, config: Dict[str, Any] = None):
        self.symbols = set(symbols or [])
        self.config = config or {}
        
        # API keys
        self.newsapi_key = self.config.get('newsapi_key')
        self.finnhub_key = self.config.get('finnhub_key')
        self.alpha_vantage_key = self.config.get('alpha_vantage_key')
        
        # Feed state
        self.is_running = False
        self.session: Optional[aiohttp.ClientSession] = None
        self.tasks: List[asyncio.Task] = []
        
        # Data storage
        self.news_cache: List[NewsItem] = []
        self.max_cache_size = self.config.get('max_cache_size', 1000)
        self.cache_duration_hours = self.config.get('cache_duration_hours', 24)
        
        # Callbacks
        self.news_callbacks: List[callable] = []
        self.data_callbacks: List[callable] = []
        
        # Configuration
        self.update_interval = self.config.get('update_interval', 300)  # 5 minutes
        self.enable_sentiment_analysis = self.config.get('enable_sentiment', True)
        self.enable_symbol_extraction = self.config.get('enable_symbol_extraction', True)
        self.filter_financial_news = self.config.get('filter_financial_news', True)
        
        # Language settings
        self.languages = self.config.get('languages', ['en'])
        self.countries = self.config.get('countries', ['us'])
        
        # Metrics
        self.news_items_processed = 0
        self.api_calls_made = 0
        self.errors_count = 0
    
    async def start(self) -> None:
        """Start the news feed"""
        if self.is_running:
            return
        
        logger.info("Starting news feed")
        
        # Create HTTP session
        self.session = aiohttp.ClientSession()
        
        # Start news fetching tasks
        self.tasks.append(asyncio.create_task(self._news_fetcher()))
        self.tasks.append(asyncio.create_task(self._cache_cleaner()))
        
        self.is_running = True
        logger.info("News feed started successfully")
    
    async def stop(self) -> None:
        """Stop the news feed"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel tasks
        for task in self.tasks:
            task.cancel()
        
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Close session
        if self.session:
            await self.session.close()
        
        logger.info("News feed stopped")
    
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to news for additional symbols"""
        new_symbols = set(symbols) - self.symbols
        if new_symbols:
            self.symbols.update(new_symbols)
            logger.info(f"Subscribed to news for {len(new_symbols)} new symbols")
    
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from news for symbols"""
        symbols_to_remove = set(symbols) & self.symbols
        if symbols_to_remove:
            self.symbols -= symbols_to_remove
            # Clean up cache
            self.news_cache = [
                item for item in self.news_cache 
                if not any(symbol in item.symbols for symbol in symbols_to_remove)
            ]
            logger.info(f"Unsubscribed from news for {len(symbols_to_remove)} symbols")
    
    async def _news_fetcher(self) -> None:
        """Main news fetching loop"""
        while self.is_running:
            try:
                # Fetch from all available sources
                news_items = []
                
                if self.newsapi_key:
                    newsapi_items = await self._fetch_newsapi()
                    news_items.extend(newsapi_items)
                
                if self.finnhub_key:
                    finnhub_items = await self._fetch_finnhub_news()
                    news_items.extend(finnhub_items)
                
                if self.alpha_vantage_key:
                    av_items = await self._fetch_alpha_vantage_news()
                    news_items.extend(av_items)
                
                # Process and filter news items
                for item in news_items:
                    await self._process_news_item(item)
                
                # Wait before next fetch
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"News fetcher error: {e}")
                self.errors_count += 1
                await asyncio.sleep(60)  # Wait on error
    
    async def _fetch_newsapi(self) -> List[NewsItem]:
        """Fetch news from NewsAPI"""
        if not self.newsapi_key or not self.session:
            return []
        
        try:
            # Build query for symbols
            query = " OR ".join(self.symbols) if self.symbols else "financial market"
            
            params = {
                'q': query,
                'apiKey': self.newsapi_key,
                'language': ','.join(self.languages),
                'sortBy': 'publishedAt',
                'pageSize': 50,
                'from': (datetime.now() - timedelta(hours=1)).isoformat()
            }
            
            url = self.SOURCES['newsapi']['url']
            self.api_calls_made += 1
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get('articles', [])
                    
                    news_items = []
                    for article in articles:
                        if self._is_relevant_news(article):
                            item = self._convert_newsapi_article(article)
                            news_items.append(item)
                    
                    return news_items
                else:
                    logger.error(f"NewsAPI error: {response.status}")
                    
        except Exception as e:
            logger.error(f"NewsAPI fetch error: {e}")
            self.errors_count += 1
        
        return []
    
    async def _fetch_finnhub_news(self) -> List[NewsItem]:
        """Fetch news from Finnhub"""
        if not self.finnhub_key or not self.session:
            return []
        
        try:
            # Finnhub provides general market news
            params = {
                'category': 'general',
                'token': self.finnhub_key,
                'minId': 0  # Get recent news
            }
            
            url = self.SOURCES['finnhub_news']['url']
            self.api_calls_made += 1
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    articles = await response.json()
                    
                    news_items = []
                    for article in articles:
                        if self._is_relevant_news(article):
                            item = self._convert_finnhub_article(article)
                            news_items.append(item)
                    
                    return news_items
                else:
                    logger.error(f"Finnhub news error: {response.status}")
                    
        except Exception as e:
            logger.error(f"Finnhub news fetch error: {e}")
            self.errors_count += 1
        
        return []
    
    async def _fetch_alpha_vantage_news(self) -> List[NewsItem]:
        """Fetch news from Alpha Vantage"""
        if not self.alpha_vantage_key or not self.session:
            return []
        
        try:
            params = {
                'function': 'NEWS_SENTIMENT',
                'apikey': self.alpha_vantage_key,
                'limit': 50
            }
            
            url = self.SOURCES['alpha_vantage_news']['url']
            self.api_calls_made += 1
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    feed = data.get('feed', [])
                    
                    news_items = []
                    for article in feed:
                        if self._is_relevant_news(article):
                            item = self._convert_alpha_vantage_article(article)
                            news_items.append(item)
                    
                    return news_items
                else:
                    logger.error(f"Alpha Vantage news error: {response.status}")
                    
        except Exception as e:
            logger.error(f"Alpha Vantage news fetch error: {e}")
            self.errors_count += 1
        
        return []
    
    def _is_relevant_news(self, article: Dict[str, Any]) -> bool:
        """Check if news article is relevant"""
        if not self.filter_financial_news:
            return True
        
        # Get article text
        title = article.get('title', '').lower()
        description = article.get('description', '').lower()
        content = article.get('content', '').lower()
        summary = article.get('summary', '').lower()
        
        text = f"{title} {description} {content} {summary}"
        
        # Check for financial keywords
        for keyword in self.FINANCIAL_KEYWORDS:
            if keyword in text:
                return True
        
        # Check for symbol mentions
        for symbol in self.symbols:
            if symbol.lower() in text:
                return True
        
        return False
    
    def _convert_newsapi_article(self, article: Dict[str, Any]) -> NewsItem:
        """Convert NewsAPI article to NewsItem"""
        return NewsItem(
            title=article.get('title', ''),
            summary=article.get('description', ''),
            url=article.get('url', ''),
            source=f"newsapi_{article.get('source', {}).get('name', 'unknown')}",
            published_at=datetime.fromisoformat(article.get('publishedAt', '').replace('Z', '+00:00')),
            symbols=self._extract_symbols(article),
            metadata={
                'author': article.get('author'),
                'urlToImage': article.get('urlToImage')
            }
        )
    
    def _convert_finnhub_article(self, article: Dict[str, Any]) -> NewsItem:
        """Convert Finnhub article to NewsItem"""
        return NewsItem(
            title=article.get('headline', ''),
            summary=article.get('summary', ''),
            url=article.get('url', ''),
            source=f"finnhub_{article.get('source', 'unknown')}",
            published_at=datetime.fromtimestamp(article.get('datetime', 0)),
            symbols=self._extract_symbols(article),
            metadata={
                'id': article.get('id'),
                'image': article.get('image'),
                'related': article.get('related')
            }
        )
    
    def _convert_alpha_vantage_article(self, article: Dict[str, Any]) -> NewsItem:
        """Convert Alpha Vantage article to NewsItem"""
        return NewsItem(
            title=article.get('title', ''),
            summary=article.get('summary', ''),
            url=article.get('url', ''),
            source=f"alphavantage_{article.get('source', 'unknown')}",
            published_at=datetime.fromisoformat(article.get('time_published', '')),
            symbols=self._extract_symbols_from_tickers(article.get('ticker_sentiment', [])),
            sentiment=self._parse_sentiment(article.get('overall_sentiment_score')),
            metadata={
                'authors': article.get('authors', []),
                'topics': article.get('topics', []),
                'overall_sentiment_label': article.get('overall_sentiment_label')
            }
        )
    
    def _extract_symbols(self, article: Dict[str, Any]) -> List[str]:
        """Extract stock symbols from article text"""
        if not self.enable_symbol_extraction:
            return []
        
        symbols = []
        text = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}"
        
        # Look for stock symbols in format $SYMBOL or (SYMBOL)
        symbol_pattern = r'\$([A-Z]{1,5})\b|\(([A-Z]{1,5})\)'
        matches = re.findall(symbol_pattern, text.upper())
        
        for match in matches:
            symbol = match[0] or match[1]
            if len(symbol) <= 5:  # Valid stock symbol length
                symbols.append(symbol)
        
        # Add symbols we're tracking if mentioned
        for symbol in self.symbols:
            if symbol.upper() in text.upper():
                symbols.append(symbol)
        
        return list(set(symbols))  # Remove duplicates
    
    def _extract_symbols_from_tickers(self, ticker_sentiment: List[Dict]) -> List[str]:
        """Extract symbols from Alpha Vantage ticker sentiment data"""
        return [item.get('ticker', '') for item in ticker_sentiment if item.get('ticker')]
    
    def _parse_sentiment(self, sentiment_score: Union[str, float]) -> Optional[float]:
        """Parse sentiment score"""
        if sentiment_score is None:
            return None
        
        try:
            return float(sentiment_score)
        except (ValueError, TypeError):
            return None
    
    async def _process_news_item(self, item: NewsItem) -> None:
        """Process and enhance news item"""
        try:
            # Analyze sentiment if not provided
            if item.sentiment is None and self.enable_sentiment_analysis:
                item.sentiment = self._analyze_sentiment(item.title, item.summary)
            
            # Calculate impact score
            item.impact_score = self._calculate_impact_score(item)
            
            # Categorize news
            item.category = self._categorize_news(item)
            
            # Add to cache
            self._add_to_cache(item)
            
            # Create DataPoint for integration
            data_point = DataPoint(
                source="news_feed",
                symbol="NEWS",
                timestamp=item.published_at,
                price=item.sentiment or 0,  # Use sentiment as "price"
                volume=int(item.impact_score * 100) if item.impact_score else 0,
                metadata={
                    'title': item.title,
                    'summary': item.summary,
                    'url': item.url,
                    'news_source': item.source,
                    'symbols': item.symbols,
                    'sentiment': item.sentiment,
                    'impact_score': item.impact_score,
                    'category': item.category
                }
            )
            
            # Notify callbacks
            for callback in self.news_callbacks:
                try:
                    await callback(item)
                except Exception as e:
                    logger.error(f"News callback error: {e}")
            
            for callback in self.data_callbacks:
                try:
                    await callback(data_point)
                except Exception as e:
                    logger.error(f"Data callback error: {e}")
            
            self.news_items_processed += 1
            
        except Exception as e:
            logger.error(f"Error processing news item: {e}")
    
    def _analyze_sentiment(self, title: str, summary: str) -> float:
        """Simple sentiment analysis based on keywords"""
        text = f"{title} {summary}".lower()
        
        positive_count = sum(1 for word in self.POSITIVE_KEYWORDS if word in text)
        negative_count = sum(1 for word in self.NEGATIVE_KEYWORDS if word in text)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        # Return score between -1 and 1
        sentiment = (positive_count - negative_count) / (positive_count + negative_count)
        return max(-1.0, min(1.0, sentiment))
    
    def _calculate_impact_score(self, item: NewsItem) -> float:
        """Calculate news impact score"""
        score = 0.0
        
        # Base score
        score += 0.3
        
        # Symbol relevance
        if item.symbols:
            score += 0.3 * min(len(item.symbols), 3) / 3
        
        # Sentiment strength
        if item.sentiment is not None:
            score += 0.2 * abs(item.sentiment)
        
        # Recency boost
        hours_old = (datetime.now() - item.published_at).total_seconds() / 3600
        if hours_old < 1:
            score += 0.2
        elif hours_old < 6:
            score += 0.1
        
        return min(1.0, score)
    
    def _categorize_news(self, item: NewsItem) -> str:
        """Categorize news item"""
        text = f"{item.title} {item.summary}".lower()
        
        categories = {
            'earnings': ['earnings', 'revenue', 'profit', 'quarter'],
            'merger': ['merger', 'acquisition', 'buyout', 'takeover'],
            'regulatory': ['sec', 'regulation', 'compliance', 'filing'],
            'economic': ['fed', 'interest rate', 'inflation', 'gdp', 'economic'],
            'market': ['market', 'trading', 'volume', 'price'],
            'corporate': ['ceo', 'management', 'board', 'executive']
        }
        
        for category, keywords in categories.items():
            if any(keyword in text for keyword in keywords):
                return category
        
        return 'general'
    
    def _add_to_cache(self, item: NewsItem) -> None:
        """Add news item to cache"""
        # Check for duplicates
        for cached_item in self.news_cache:
            if cached_item.url == item.url or cached_item.title == item.title:
                return
        
        # Add to cache
        self.news_cache.append(item)
        
        # Maintain cache size
        if len(self.news_cache) > self.max_cache_size:
            self.news_cache = self.news_cache[-self.max_cache_size:]
    
    async def _cache_cleaner(self) -> None:
        """Clean old items from cache"""
        while self.is_running:
            try:
                cutoff_time = datetime.now() - timedelta(hours=self.cache_duration_hours)
                self.news_cache = [
                    item for item in self.news_cache 
                    if item.published_at > cutoff_time
                ]
                
                await asyncio.sleep(3600)  # Clean every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleaner error: {e}")
    
    def add_news_callback(self, callback: callable) -> None:
        """Add news callback"""
        self.news_callbacks.append(callback)
    
    def add_data_callback(self, callback: callable) -> None:
        """Add data callback"""
        self.data_callbacks.append(callback)
    
    def get_recent_news(self, hours: int = 24, symbols: List[str] = None) -> List[NewsItem]:
        """Get recent news items"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_news = [
            item for item in self.news_cache
            if item.published_at > cutoff_time
        ]
        
        if symbols:
            symbol_set = set(s.upper() for s in symbols)
            filtered_news = [
                item for item in filtered_news
                if any(symbol.upper() in symbol_set for symbol in item.symbols)
            ]
        
        return sorted(filtered_news, key=lambda x: x.published_at, reverse=True)
    
    def get_sentiment_summary(self, symbol: str = None, hours: int = 24) -> Dict[str, Any]:
        """Get sentiment summary for symbol or overall"""
        recent_news = self.get_recent_news(hours, [symbol] if symbol else None)
        
        if not recent_news:
            return {'count': 0, 'avg_sentiment': 0, 'positive': 0, 'negative': 0, 'neutral': 0}
        
        sentiments = [item.sentiment for item in recent_news if item.sentiment is not None]
        
        if not sentiments:
            return {'count': len(recent_news), 'avg_sentiment': 0, 'positive': 0, 'negative': 0, 'neutral': len(recent_news)}
        
        positive = sum(1 for s in sentiments if s > 0.1)
        negative = sum(1 for s in sentiments if s < -0.1)
        neutral = len(sentiments) - positive - negative
        
        return {
            'count': len(recent_news),
            'avg_sentiment': sum(sentiments) / len(sentiments),
            'positive': positive,
            'negative': negative,
            'neutral': neutral
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get news feed metrics"""
        return {
            'feed_type': 'news',
            'is_running': self.is_running,
            'symbols_count': len(self.symbols),
            'cache_size': len(self.news_cache),
            'news_items_processed': self.news_items_processed,
            'api_calls_made': self.api_calls_made,
            'errors_count': self.errors_count,
            'update_interval': self.update_interval,
            'sources_configured': len([k for k, v in {
                'newsapi': self.newsapi_key,
                'finnhub': self.finnhub_key,
                'alpha_vantage': self.alpha_vantage_key
            }.items() if v])
        }