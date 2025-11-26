"""
Finnhub Provider
Integrates with Finnhub API for real-time market data and news
"""

import asyncio
import aiohttp
import websockets
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import os

from ..news_aggregator import NewsProvider, UnifiedNewsItem

logger = logging.getLogger(__name__)


class FinnhubProvider(NewsProvider):
    """Finnhub API provider for market data and news"""
    
    BASE_URL = "https://finnhub.io/api/v1"
    WS_URL = "wss://ws.finnhub.io"
    
    # Source reliability for Finnhub news sources
    SOURCE_RELIABILITY = {
        'reuters': 0.95,
        'bloomberg': 0.95,
        'marketwatch': 0.85,
        'cnbc': 0.85,
        'yahoo': 0.75,
        'seekingalpha': 0.70,
        'thefly': 0.65,
        'default': 0.60
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get('api_key') or os.environ.get('FINNHUB_API_KEY')
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        
        if not self.api_key:
            raise ValueError("Finnhub API key is required")
        
        # Plan configuration  
        self.plan = config.get('plan', 'free')  # free, professional, enterprise
        self.rate_limits = {
            'free': {'calls_per_minute': 60, 'websocket_connections': 50},
            'professional': {'calls_per_minute': 300, 'websocket_connections': 500},
            'enterprise': {'calls_per_minute': 1000, 'websocket_connections': 1000}
        }
        
        self.current_limit = self.rate_limits.get(self.plan, self.rate_limits['free'])
        
        # Request tracking
        self.requests_made = 0
        self.minute_start = datetime.now()
        self.last_request_time = None
        self.session: Optional[aiohttp.ClientSession] = None
        
        # WebSocket configuration
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.websocket_subscriptions: set = set()
        self.websocket_callbacks: List[Callable] = []
        self.websocket_connected = False
        
        logger.info(f"Initialized Finnhub provider (plan: {self.plan})")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self.session
    
    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Make API request with rate limiting and error handling"""
        # Rate limiting
        await self._rate_limit()
        
        # Add API key to params
        if params is None:
            params = {}
        params['token'] = self.api_key
        
        session = await self._get_session()
        url = f"{self.BASE_URL}/{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                async with session.get(url, params=params) as response:
                    self.requests_made += 1
                    self.last_request_time = datetime.now()
                    
                    if response.status == 200:
                        data = await response.json()
                        return data
                    
                    elif response.status == 401:
                        logger.error("Finnhub authentication failed - check API key")
                        return None
                    
                    elif response.status == 429:
                        logger.warning("Finnhub rate limit exceeded")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(60)  # Wait 1 minute
                        continue
                    
                    else:
                        logger.error(f"Finnhub HTTP error {response.status}: {await response.text()}")
                        return None
                        
            except asyncio.TimeoutError:
                logger.warning(f"Finnhub timeout (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                continue
                
            except Exception as e:
                logger.error(f"Finnhub request error: {e} (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                continue
        
        logger.error("All Finnhub retry attempts failed")
        return None
    
    async def _rate_limit(self):
        """Implement per-minute rate limiting"""
        now = datetime.now()
        
        # Reset counter if minute has passed
        if (now - self.minute_start).total_seconds() >= 60:
            self.requests_made = 0
            self.minute_start = now
        
        # Check if we need to wait
        if self.requests_made >= self.current_limit['calls_per_minute']:
            wait_time = 60 - (now - self.minute_start).total_seconds()
            if wait_time > 0:
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self.requests_made = 0
                self.minute_start = datetime.now()
    
    async def fetch_news(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        limit: int = 100
    ) -> List[UnifiedNewsItem]:
        """
        Fetch news from Finnhub
        
        Args:
            symbols: Stock symbols to fetch news for
            start_date: Start date for news search
            end_date: End date for news search
            limit: Maximum number of items to return
            
        Returns:
            List of unified news items
        """
        try:
            all_news = []
            
            # Fetch company news for each symbol
            for symbol in symbols[:10]:  # Limit to avoid rate limits
                try:
                    # Convert dates to required format (YYYY-MM-DD)
                    from_date = start_date.strftime('%Y-%m-%d')
                    to_date = end_date.strftime('%Y-%m-%d')
                    
                    # Fetch company-specific news
                    params = {
                        'symbol': symbol,
                        'from': from_date,
                        'to': to_date
                    }
                    
                    logger.debug(f"Fetching Finnhub news for {symbol}")
                    
                    data = await self._make_request('company-news', params)
                    if data and isinstance(data, list):
                        symbol_news = []
                        for article in data[:20]:  # Limit per symbol
                            item = self._convert_to_unified(article, symbol)
                            if item and self._is_within_date_range(item, start_date, end_date):
                                symbol_news.append(item)
                        
                        all_news.extend(symbol_news)
                        logger.debug(f"Processed {len(symbol_news)} articles for {symbol}")
                    
                    # Small delay between symbol requests
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Error fetching news for {symbol}: {e}")
                    continue
            
            # Fetch general market news if no symbols or additional news needed
            if not symbols or len(all_news) < limit // 2:
                try:
                    general_data = await self._make_request('news', {'category': 'general'})
                    if general_data and isinstance(general_data, list):
                        for article in general_data[:50]:  # Limit general news
                            item = self._convert_to_unified(article)
                            if item and self._is_within_date_range(item, start_date, end_date):
                                # Check if relevant to our symbols
                                if not symbols or self._is_relevant_to_symbols(item, symbols):
                                    all_news.append(item)
                    
                except Exception as e:
                    logger.warning(f"Error fetching general news: {e}")
            
            # Remove duplicates and sort by date
            unique_news = self._deduplicate_news(all_news)
            unique_news.sort(key=lambda x: x.published_at, reverse=True)
            
            logger.info(f"Processed {len(unique_news)} unique news items from Finnhub")
            return unique_news[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching Finnhub news: {e}")
            return []
    
    def _convert_to_unified(self, article: Dict[str, Any], primary_symbol: str = None) -> Optional[UnifiedNewsItem]:
        """Convert Finnhub article to unified format"""
        try:
            headline = article.get('headline', '')
            summary = article.get('summary', '')
            url = article.get('url', '')
            
            if not headline or not url:
                return None
            
            # Parse timestamp
            timestamp = article.get('datetime', 0)
            if timestamp:
                published_at = datetime.fromtimestamp(timestamp)
            else:
                published_at = datetime.now()
            
            # Extract source
            source = article.get('source', 'unknown').lower()
            source_reliability = self.SOURCE_RELIABILITY.get(source, self.SOURCE_RELIABILITY['default'])
            
            # Extract symbols
            symbols = []
            if primary_symbol:
                symbols.append(primary_symbol.upper())
            
            # Look for additional symbols in related field
            related = article.get('related', '')
            if related:
                symbols.extend([s.strip().upper() for s in related.split(',') if s.strip()])
            
            # Remove duplicates
            symbols = list(set(symbols))
            
            # Basic sentiment analysis
            sentiment = self._analyze_sentiment_basic(headline, summary)
            
            # Categorize
            categories = self._categorize_finnhub_article(headline, summary)
            
            return UnifiedNewsItem(
                id=f"finnhub_{article.get('id', str(hash(url)))}",
                title=headline,
                summary=summary,
                content=None,  # Finnhub doesn't provide full content
                url=url,
                source=f"finnhub_{source}",
                source_reliability=source_reliability,
                published_at=published_at,
                symbols=symbols,
                sentiment=sentiment,
                relevance_scores={symbol: 0.8 for symbol in symbols},  # High relevance for company news
                categories=categories,
                language='en',
                metadata={
                    'image': article.get('image'),
                    'related': article.get('related'),
                    'finnhub_id': article.get('id')
                }
            )
            
        except Exception as e:
            logger.error(f"Error converting Finnhub article: {e}")
            return None
    
    def _analyze_sentiment_basic(self, headline: str, summary: str) -> Optional[float]:
        """Basic sentiment analysis"""
        positive_words = [
            'beat', 'exceed', 'surge', 'rally', 'gain', 'rise', 'up',
            'strong', 'growth', 'positive', 'bullish', 'upgrade'
        ]
        
        negative_words = [
            'miss', 'fall', 'drop', 'decline', 'loss', 'down', 'weak',
            'negative', 'bearish', 'downgrade', 'concern', 'risk'
        ]
        
        text = f"{headline} {summary}".lower()
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / total
        return max(-1.0, min(1.0, sentiment))
    
    def _categorize_finnhub_article(self, headline: str, summary: str) -> List[str]:
        """Categorize Finnhub article"""
        text = f"{headline} {summary}".lower()
        categories = []
        
        category_keywords = {
            'earnings': ['earnings', 'eps', 'revenue', 'profit', 'quarter'],
            'merger': ['merger', 'acquisition', 'deal', 'buyout'],
            'analyst': ['upgrade', 'downgrade', 'target', 'rating', 'analyst'],
            'insider': ['insider', 'ceo', 'executive', 'director'],
            'regulatory': ['sec', 'fda', 'regulatory', 'approval'],
            'market': ['market', 'trading', 'volume', 'index']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in text for keyword in keywords):
                categories.append(category)
        
        return categories if categories else ['general']
    
    def _is_within_date_range(self, item: UnifiedNewsItem, start_date: datetime, end_date: datetime) -> bool:
        """Check if news item is within date range"""
        return start_date <= item.published_at <= end_date
    
    def _is_relevant_to_symbols(self, item: UnifiedNewsItem, symbols: List[str]) -> bool:
        """Check if news item is relevant to given symbols"""
        if item.symbols:
            return any(symbol.upper() in [s.upper() for s in item.symbols] for symbol in symbols)
        
        # Check text content
        text = f"{item.title} {item.summary}".upper()
        return any(symbol.upper() in text for symbol in symbols)
    
    def _deduplicate_news(self, news_items: List[UnifiedNewsItem]) -> List[UnifiedNewsItem]:
        """Remove duplicate news items"""
        seen_urls = set()
        seen_titles = set()
        unique_items = []
        
        for item in news_items:
            if item.url in seen_urls or item.title in seen_titles:
                continue
            
            seen_urls.add(item.url)
            seen_titles.add(item.title)
            unique_items.append(item)
        
        return unique_items
    
    async def get_sentiment(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> Dict[str, Any]:
        """Get sentiment summary for a specific symbol"""
        try:
            # Fetch recent news
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=lookback_hours)
            
            news_items = await self.fetch_news([symbol], start_date, end_date, limit=50)
            
            if not news_items:
                return {
                    'symbol': symbol,
                    'sentiment_score': 0.0,
                    'sentiment_label': 'neutral',
                    'confidence': 0.0,
                    'article_count': 0,
                    'buzz_score': 0.0
                }
            
            # Calculate sentiment metrics
            sentiments = [item.sentiment for item in news_items if item.sentiment is not None]
            
            if not sentiments:
                avg_sentiment = 0.0
                confidence = 0.0
            else:
                avg_sentiment = sum(sentiments) / len(sentiments)
                sentiment_variance = sum((s - avg_sentiment) ** 2 for s in sentiments) / len(sentiments)
                confidence = max(0.0, 1.0 - (sentiment_variance ** 0.5))
            
            # Calculate buzz score (article frequency)
            hours_covered = (end_date - start_date).total_seconds() / 3600
            buzz_score = len(news_items) / max(1, hours_covered) * 24  # Articles per day
            
            # Determine sentiment label
            if avg_sentiment > 0.1:
                sentiment_label = 'bullish'
            elif avg_sentiment < -0.1:
                sentiment_label = 'bearish'
            else:
                sentiment_label = 'neutral'
            
            return {
                'symbol': symbol,
                'sentiment_score': avg_sentiment,
                'sentiment_label': sentiment_label,
                'confidence': confidence,
                'article_count': len(news_items),
                'buzz_score': min(buzz_score, 10.0),  # Cap at 10
                'time_range_hours': lookback_hours,
                'sources': list(set(item.source for item in news_items))
            }
            
        except Exception as e:
            logger.error(f"Error getting Finnhub sentiment for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.0,
                'article_count': 0,
                'buzz_score': 0.0
            }
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get real-time market data for symbol"""
        try:
            data = await self._make_request('quote', {'symbol': symbol})
            if not data:
                return {}
            
            return {
                'symbol': symbol,
                'current_price': data.get('c'),
                'change': data.get('d'),
                'percent_change': data.get('dp'),
                'high': data.get('h'),
                'low': data.get('l'),
                'open': data.get('o'),
                'previous_close': data.get('pc'),
                'timestamp': datetime.fromtimestamp(data.get('t', 0)).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return {}
    
    async def start_websocket(self):
        """Start WebSocket connection for real-time data"""
        try:
            ws_url = f"{self.WS_URL}?token={self.api_key}"
            self.websocket = await websockets.connect(ws_url)
            self.websocket_connected = True
            
            # Start message handler
            asyncio.create_task(self._websocket_handler())
            
            logger.info("Finnhub WebSocket connected")
            
        except Exception as e:
            logger.error(f"Error starting Finnhub WebSocket: {e}")
            self.websocket_connected = False
    
    async def _websocket_handler(self):
        """Handle WebSocket messages"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                
                # Process different message types
                if data.get('type') == 'trade':
                    await self._handle_trade_message(data)
                elif data.get('type') == 'ping':
                    # Respond to ping
                    await self.websocket.send(json.dumps({'type': 'pong'}))
                
                # Notify callbacks
                for callback in self.websocket_callbacks:
                    try:
                        await callback(data)
                    except Exception as e:
                        logger.error(f"WebSocket callback error: {e}")
                        
        except websockets.exceptions.ConnectionClosed:
            logger.info("Finnhub WebSocket connection closed")
            self.websocket_connected = False
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
            self.websocket_connected = False
    
    async def _handle_trade_message(self, data: Dict[str, Any]):
        """Handle trade messages from WebSocket"""
        trades = data.get('data', [])
        for trade in trades:
            symbol = trade.get('s')
            price = trade.get('p')
            volume = trade.get('v')
            timestamp = trade.get('t')
            
            logger.debug(f"Trade: {symbol} @ {price} vol: {volume}")
    
    async def subscribe_to_trades(self, symbols: List[str]):
        """Subscribe to real-time trades for symbols"""
        if not self.websocket_connected:
            await self.start_websocket()
        
        for symbol in symbols:
            if symbol not in self.websocket_subscriptions:
                message = {
                    'type': 'subscribe',
                    'symbol': symbol
                }
                await self.websocket.send(json.dumps(message))
                self.websocket_subscriptions.add(symbol)
                logger.debug(f"Subscribed to {symbol} trades")
    
    def add_websocket_callback(self, callback: Callable):
        """Add callback for WebSocket messages"""
        self.websocket_callbacks.append(callback)
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get current rate limit status"""
        now = datetime.now()
        time_until_reset = 60 - (now - self.minute_start).total_seconds()
        
        return {
            'plan': self.plan,
            'calls_per_minute': self.current_limit['calls_per_minute'],
            'calls_made_this_minute': self.requests_made,
            'calls_remaining': max(0, self.current_limit['calls_per_minute'] - self.requests_made),
            'reset_in_seconds': max(0, time_until_reset),
            'websocket_connected': self.websocket_connected,
            'websocket_subscriptions': len(self.websocket_subscriptions),
            'last_request_time': self.last_request_time.isoformat() if self.last_request_time else None
        }
    
    async def health_check(self) -> bool:
        """Check if Finnhub API is accessible"""
        try:
            # Make a simple quote request
            data = await self._make_request('quote', {'symbol': 'AAPL'})
            return data is not None and 'c' in data
            
        except Exception as e:
            logger.error(f"Finnhub health check failed: {e}")
            return False
    
    async def close(self):
        """Close connections and cleanup"""
        try:
            # Close WebSocket
            if self.websocket and not self.websocket.closed:
                await self.websocket.close()
                self.websocket_connected = False
            
            # Close HTTP session
            if self.session and not self.session.closed:
                await self.session.close()
                self.session = None
            
            logger.info("Finnhub provider closed")
            
        except Exception as e:
            logger.error(f"Error closing Finnhub provider: {e}")