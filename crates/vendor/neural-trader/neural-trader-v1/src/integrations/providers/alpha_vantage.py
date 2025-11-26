"""
Alpha Vantage News Provider
Integrates with Alpha Vantage API for financial news and sentiment data
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
import dateutil.parser

from ..news_aggregator import NewsProvider, UnifiedNewsItem

logger = logging.getLogger(__name__)


class AlphaVantageNewsProvider(NewsProvider):
    """Alpha Vantage news and sentiment provider"""
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    # Source reliability scores
    SOURCE_RELIABILITY = {
        'reuters': 0.95,
        'bloomberg': 0.95,
        'wall street journal': 0.90,
        'financial times': 0.90,
        'cnbc': 0.85,
        'marketwatch': 0.80,
        'yahoo finance': 0.75,
        'seeking alpha': 0.70,
        'default': 0.60
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get('api_key') or os.environ.get('ALPHA_VANTAGE_API_KEY')
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required")
        
        # Rate limiting based on tier
        self.tier = config.get('tier', 'free')
        self.rate_limits = {
            'free': {'calls_per_minute': 5, 'daily_limit': 500},
            'starter': {'calls_per_minute': 30, 'daily_limit': 5000},
            'professional': {'calls_per_minute': 60, 'daily_limit': 10000},
            'enterprise': {'calls_per_minute': 120, 'daily_limit': 25000}
        }
        
        self.current_limit = self.rate_limits.get(self.tier, self.rate_limits['free'])
        
        # Request tracking
        self.requests_made = 0
        self.daily_requests = 0
        self.last_request_time = None
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"Initialized Alpha Vantage provider (tier: {self.tier})")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self.session
    
    async def _make_request(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make API request with rate limiting and error handling"""
        # Add API key
        params['apikey'] = self.api_key
        
        # Rate limiting
        await self._rate_limit()
        
        session = await self._get_session()
        
        for attempt in range(self.max_retries):
            try:
                async with session.get(self.BASE_URL, params=params) as response:
                    self.requests_made += 1
                    self.daily_requests += 1
                    self.last_request_time = datetime.now()
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        # Check for API error messages
                        if 'Error Message' in data:
                            logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                            return None
                        
                        if 'Note' in data:
                            logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                            await asyncio.sleep(60)  # Wait a minute on rate limit
                            continue
                        
                        return data
                    
                    elif response.status == 429:
                        # Rate limit exceeded
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limited, waiting {wait_time}s (attempt {attempt + 1})")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    else:
                        logger.error(f"HTTP error {response.status}: {await response.text()}")
                        return None
                        
            except asyncio.TimeoutError:
                logger.warning(f"Request timeout (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                continue
                
            except Exception as e:
                logger.error(f"Request error: {e} (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                continue
        
        logger.error("All retry attempts failed")
        return None
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        if self.last_request_time:
            time_since_last = (datetime.now() - self.last_request_time).total_seconds()
            min_interval = 60 / self.current_limit['calls_per_minute']
            
            if time_since_last < min_interval:
                wait_time = min_interval - time_since_last
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
    
    async def fetch_news(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        limit: int = 100
    ) -> List[UnifiedNewsItem]:
        """
        Fetch news from Alpha Vantage NEWS_SENTIMENT endpoint
        
        Args:
            symbols: Stock symbols to fetch news for
            start_date: Start date for news search
            end_date: End date for news search
            limit: Maximum number of items to return
            
        Returns:
            List of unified news items
        """
        try:
            # Alpha Vantage doesn't support date ranges in free tier
            # Use tickers parameter for symbol filtering
            tickers = ','.join(symbols[:10])  # Limit to 10 symbols
            
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': tickers,
                'limit': min(limit, 1000),  # API max is 1000
                'sort': 'LATEST'
            }
            
            # Add time_from/time_to for premium tiers
            if self.tier != 'free':
                params['time_from'] = start_date.strftime('%Y%m%dT%H%M')
                params['time_to'] = end_date.strftime('%Y%m%dT%H%M')
            
            logger.debug(f"Fetching Alpha Vantage news for symbols: {symbols}")
            
            data = await self._make_request(params)
            if not data or 'feed' not in data:
                logger.warning("No news data received from Alpha Vantage")
                return []
            
            articles = data['feed']
            logger.debug(f"Received {len(articles)} articles from Alpha Vantage")
            
            # Convert to unified format
            news_items = []
            for article in articles:
                try:
                    # Filter by date if needed (for free tier)
                    published_date = dateutil.parser.parse(article.get('time_published', ''))
                    if published_date < start_date or published_date > end_date:
                        continue
                    
                    # Filter by symbol relevance
                    article_symbols = self._extract_symbols_from_tickers(
                        article.get('ticker_sentiment', [])
                    )
                    
                    # Check if any of our target symbols are mentioned
                    if symbols and not any(symbol.upper() in [s.upper() for s in article_symbols] for symbol in symbols):
                        continue
                    
                    item = self._convert_to_unified(article)
                    if item:
                        news_items.append(item)
                        
                except Exception as e:
                    logger.warning(f"Error processing article: {e}")
                    continue
            
            logger.info(f"Processed {len(news_items)} relevant news items from Alpha Vantage")
            return news_items[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage news: {e}")
            return []
    
    def _convert_to_unified(self, article: Dict[str, Any]) -> Optional[UnifiedNewsItem]:
        """Convert Alpha Vantage article to unified format"""
        try:
            # Extract basic information
            title = article.get('title', '')
            summary = article.get('summary', '')
            url = article.get('url', '')
            
            if not title or not url:
                return None
            
            # Parse publication date
            try:
                published_at = dateutil.parser.parse(article.get('time_published', ''))
            except:
                logger.warning(f"Could not parse date: {article.get('time_published')}")
                published_at = datetime.now()
            
            # Extract source and calculate reliability
            source = article.get('source', 'unknown').lower()
            source_reliability = self.SOURCE_RELIABILITY.get(source, self.SOURCE_RELIABILITY['default'])
            
            # Extract symbols and relevance scores
            ticker_sentiment = article.get('ticker_sentiment', [])
            symbols = self._extract_symbols_from_tickers(ticker_sentiment)
            relevance_scores = self._extract_relevance_scores(ticker_sentiment)
            
            # Convert sentiment score
            sentiment = self._normalize_sentiment(article.get('overall_sentiment_score'))
            
            # Extract categories from topics
            categories = article.get('topics', [])
            if isinstance(categories, list):
                categories = [topic.get('topic', '') if isinstance(topic, dict) else str(topic) 
                            for topic in categories]
            
            # Create unified item
            return UnifiedNewsItem(
                id=f"av_{self._generate_id(article)}",
                title=title,
                summary=summary,
                content=None,  # AV doesn't provide full content
                url=url,
                source=f"alphavantage_{source}",
                source_reliability=source_reliability,
                published_at=published_at,
                symbols=symbols,
                sentiment=sentiment,
                relevance_scores=relevance_scores,
                categories=categories,
                language='en',
                metadata={
                    'authors': article.get('authors', []),
                    'overall_sentiment_label': article.get('overall_sentiment_label'),
                    'topics': article.get('topics', []),
                    'banner_image': article.get('banner_image'),
                    'source_domain': article.get('source_domain')
                }
            )
            
        except Exception as e:
            logger.error(f"Error converting Alpha Vantage article: {e}")
            return None
    
    def _extract_symbols_from_tickers(self, ticker_sentiment: List[Dict]) -> List[str]:
        """Extract stock symbols from ticker sentiment data"""
        symbols = []
        for item in ticker_sentiment:
            ticker = item.get('ticker', '').strip()
            if ticker and len(ticker) <= 5:  # Valid stock symbol length
                symbols.append(ticker.upper())
        return list(set(symbols))
    
    def _extract_relevance_scores(self, ticker_sentiment: List[Dict]) -> Dict[str, float]:
        """Extract relevance scores for each symbol"""
        scores = {}
        for item in ticker_sentiment:
            ticker = item.get('ticker', '').strip()
            if ticker:
                # Use relevance score if available, otherwise use sentiment strength
                relevance = item.get('relevance_score')
                if relevance is not None:
                    try:
                        scores[ticker.upper()] = float(relevance)
                    except:
                        pass
        return scores
    
    def _normalize_sentiment(self, sentiment_score: Any) -> Optional[float]:
        """
        Normalize Alpha Vantage sentiment score to -1 to 1 range
        Alpha Vantage returns scores from -1 to 1, so no conversion needed
        """
        if sentiment_score is None:
            return None
        
        try:
            score = float(sentiment_score)
            return max(-1.0, min(1.0, score))
        except (ValueError, TypeError):
            return None
    
    def _generate_id(self, article: Dict[str, Any]) -> str:
        """Generate unique ID for article"""
        # Use URL if available
        url = article.get('url', '')
        if url:
            return url.split('/')[-1][:20]
        
        # Fallback to title hash
        title = article.get('title', '')
        return str(hash(title))[:16]
    
    async def get_sentiment(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> Dict[str, Any]:
        """Get sentiment summary for a specific symbol"""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=lookback_hours)
            
            # Fetch news for the symbol
            news_items = await self.fetch_news([symbol], start_date, end_date, limit=50)
            
            if not news_items:
                return {
                    'symbol': symbol,
                    'sentiment_score': 0.0,
                    'sentiment_label': 'neutral',
                    'confidence': 0.0,
                    'article_count': 0
                }
            
            # Calculate aggregated sentiment
            sentiments = [item.sentiment for item in news_items if item.sentiment is not None]
            
            if not sentiments:
                return {
                    'symbol': symbol,
                    'sentiment_score': 0.0,
                    'sentiment_label': 'neutral',
                    'confidence': 0.0,
                    'article_count': len(news_items)
                }
            
            avg_sentiment = sum(sentiments) / len(sentiments)
            
            # Determine sentiment label
            if avg_sentiment > 0.1:
                sentiment_label = 'bullish'
            elif avg_sentiment < -0.1:
                sentiment_label = 'bearish'
            else:
                sentiment_label = 'neutral'
            
            # Calculate confidence based on consistency
            sentiment_std = (sum((s - avg_sentiment) ** 2 for s in sentiments) / len(sentiments)) ** 0.5
            confidence = max(0.0, 1.0 - sentiment_std)
            
            return {
                'symbol': symbol,
                'sentiment_score': avg_sentiment,
                'sentiment_label': sentiment_label,
                'confidence': confidence,
                'article_count': len(news_items),
                'time_range_hours': lookback_hours
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.0,
                'article_count': 0
            }
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get current rate limit status"""
        calls_remaining = self.current_limit['calls_per_minute'] - (
            self.requests_made % self.current_limit['calls_per_minute']
        )
        
        daily_remaining = self.current_limit['daily_limit'] - self.daily_requests
        
        return {
            'tier': self.tier,
            'calls_per_minute': self.current_limit['calls_per_minute'],
            'calls_remaining_this_minute': max(0, calls_remaining),
            'daily_limit': self.current_limit['daily_limit'],
            'daily_remaining': max(0, daily_remaining),
            'requests_made_today': self.daily_requests,
            'last_request_time': self.last_request_time.isoformat() if self.last_request_time else None
        }
    
    async def health_check(self) -> bool:
        """Check if Alpha Vantage API is accessible"""
        try:
            # Make a simple quote request
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': 'AAPL'
            }
            
            data = await self._make_request(params)
            return data is not None and 'Global Quote' in data
            
        except Exception as e:
            logger.error(f"Alpha Vantage health check failed: {e}")
            return False
    
    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None