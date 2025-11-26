"""
NewsAPI Provider
Integrates with NewsAPI.org for comprehensive news coverage
"""

import asyncio
import aiohttp
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
import dateutil.parser
import hashlib

from ..news_aggregator import NewsProvider, UnifiedNewsItem

logger = logging.getLogger(__name__)


class NewsAPIProvider(NewsProvider):
    """NewsAPI.org news provider"""
    
    BASE_URL = "https://newsapi.org/v2"
    
    # Financial news sources prioritized for reliability
    FINANCIAL_SOURCES = [
        'bloomberg', 'reuters', 'financial-times', 'wall-street-journal',
        'cnbc', 'marketwatch', 'fortune', 'business-insider',
        'the-economist', 'forbes', 'cnn-business', 'bbc-news'
    ]
    
    # Source reliability mapping
    SOURCE_RELIABILITY = {
        'bloomberg': 0.95,
        'reuters': 0.95,
        'financial-times': 0.92,
        'wall-street-journal': 0.90,
        'the-economist': 0.88,
        'cnbc': 0.85,
        'bbc-news': 0.85,
        'cnn-business': 0.80,
        'marketwatch': 0.80,
        'fortune': 0.78,
        'business-insider': 0.75,
        'forbes': 0.75,
        'yahoo-news': 0.70,
        'default': 0.60
    }
    
    # Financial keywords for relevance filtering
    FINANCIAL_KEYWORDS = [
        'earnings', 'revenue', 'profit', 'loss', 'dividend', 'merger',
        'acquisition', 'ipo', 'stock', 'shares', 'market', 'trading',
        'financial', 'quarter', 'guidance', 'forecast', 'outlook',
        'sec filing', 'regulations', 'federal reserve', 'interest rate',
        'inflation', 'gdp', 'unemployment', 'economic', 'fiscal'
    ]
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get('api_key') or os.environ.get('NEWSAPI_API_KEY')
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        
        if not self.api_key:
            raise ValueError("NewsAPI API key is required")
        
        # Plan configuration
        self.plan = config.get('plan', 'developer')  # developer, business, enterprise
        self.rate_limits = {
            'developer': {'requests_per_day': 100, 'results_per_request': 100},
            'business': {'requests_per_day': 250000, 'results_per_request': 100},
            'enterprise': {'requests_per_day': 1000000, 'results_per_request': 100}
        }
        
        self.current_limit = self.rate_limits.get(self.plan, self.rate_limits['developer'])
        
        # Configuration options
        self.languages = config.get('languages', ['en'])
        self.countries = config.get('countries', ['us'])
        self.enable_content_filtering = config.get('enable_content_filtering', True)
        self.min_article_length = config.get('min_article_length', 100)
        
        # Request tracking
        self.requests_made_today = 0
        self.last_request_time = None
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"Initialized NewsAPI provider (plan: {self.plan})")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            headers = {
                'X-API-Key': self.api_key,
                'User-Agent': 'AI-News-Trader/1.0'
            }
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self.session
    
    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make API request with error handling"""
        # Check daily limits
        if self.requests_made_today >= self.current_limit['requests_per_day']:
            logger.warning("Daily request limit reached for NewsAPI")
            return None
        
        session = await self._get_session()
        url = f"{self.BASE_URL}/{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                async with session.get(url, params=params) as response:
                    self.requests_made_today += 1
                    self.last_request_time = datetime.now()
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('status') == 'ok':
                            return data
                        else:
                            logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                            return None
                    
                    elif response.status == 401:
                        logger.error("NewsAPI authentication failed - check API key")
                        return None
                    
                    elif response.status == 429:
                        logger.warning("NewsAPI rate limit exceeded")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(60)  # Wait 1 minute
                        continue
                    
                    elif response.status == 426:
                        logger.error("NewsAPI upgrade required for this request")
                        return None
                    
                    else:
                        logger.error(f"NewsAPI HTTP error {response.status}: {await response.text()}")
                        return None
                        
            except asyncio.TimeoutError:
                logger.warning(f"NewsAPI timeout (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                continue
                
            except Exception as e:
                logger.error(f"NewsAPI request error: {e} (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                continue
        
        logger.error("All NewsAPI retry attempts failed")
        return None
    
    async def fetch_news(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        limit: int = 100
    ) -> List[UnifiedNewsItem]:
        """
        Fetch news from NewsAPI
        
        Args:
            symbols: Stock symbols to search for
            start_date: Start date for news search
            end_date: End date for news search  
            limit: Maximum number of items to return
            
        Returns:
            List of unified news items
        """
        try:
            # Build search query
            if symbols:
                # Create query with stock symbols and financial context
                symbol_queries = []
                for symbol in symbols[:5]:  # Limit to 5 symbols to avoid query length issues
                    # Search for symbol in various contexts
                    symbol_queries.append(f'("{symbol}" OR "${symbol}" OR "{symbol} stock")')
                
                base_query = " OR ".join(symbol_queries)
                
                # Add financial context
                query = f"({base_query}) AND (stock OR shares OR earnings OR financial OR market)"
            else:
                # General financial news query
                query = "(stock market OR financial OR earnings OR trading OR shares)"
            
            # Use business sources for better financial coverage
            sources = ','.join(self.FINANCIAL_SOURCES[:20])  # API limit is 20 sources
            
            params = {
                'q': query,
                'sources': sources,
                'language': ','.join(self.languages),
                'sortBy': 'publishedAt',
                'pageSize': min(limit, self.current_limit['results_per_request']),
                'from': start_date.isoformat(),
                'to': end_date.isoformat()
            }
            
            logger.debug(f"Fetching NewsAPI articles with query: {query[:100]}...")
            
            data = await self._make_request('everything', params)
            if not data or 'articles' not in data:
                logger.warning("No articles received from NewsAPI")
                return []
            
            articles = data['articles']
            total_results = data.get('totalResults', 0)
            
            logger.debug(f"Received {len(articles)} articles from NewsAPI (total: {total_results})")
            
            # Convert to unified format
            news_items = []
            for article in articles:
                try:
                    # Filter by relevance and quality
                    if not self._is_relevant_article(article, symbols):
                        continue
                    
                    item = self._convert_to_unified(article, symbols)
                    if item:
                        news_items.append(item)
                        
                except Exception as e:
                    logger.warning(f"Error processing NewsAPI article: {e}")
                    continue
            
            logger.info(f"Processed {len(news_items)} relevant news items from NewsAPI")
            return news_items
            
        except Exception as e:
            logger.error(f"Error fetching NewsAPI news: {e}")
            return []
    
    def _is_relevant_article(self, article: Dict[str, Any], symbols: List[str]) -> bool:
        """Check if article is relevant for financial analysis"""
        title = article.get('title', '').lower()
        description = article.get('description', '').lower()
        content = article.get('content', '').lower()
        
        if not title and not description:
            return False
        
        # Check minimum content length
        full_text = f"{title} {description} {content}"
        if len(full_text) < self.min_article_length:
            return False
        
        # Check for symbol mentions
        if symbols:
            text_upper = full_text.upper()
            for symbol in symbols:
                if symbol.upper() in text_upper:
                    return True
        
        # Check for financial keywords if no symbols or no symbol matches
        if self.enable_content_filtering:
            for keyword in self.FINANCIAL_KEYWORDS:
                if keyword in full_text:
                    return True
            return False  # No financial keywords found
        
        return True
    
    def _convert_to_unified(
        self, 
        article: Dict[str, Any], 
        target_symbols: List[str] = None
    ) -> Optional[UnifiedNewsItem]:
        """Convert NewsAPI article to unified format"""
        try:
            title = article.get('title', '')
            description = article.get('description', '')
            url = article.get('url', '')
            
            if not title or not url:
                return None
            
            # Parse publication date
            try:
                published_at = dateutil.parser.parse(article.get('publishedAt', ''))
            except:
                logger.warning(f"Could not parse date: {article.get('publishedAt')}")
                published_at = datetime.now()
            
            # Extract source information
            source_info = article.get('source', {})
            source_name = source_info.get('name', 'unknown').lower()
            source_id = source_info.get('id', source_name)
            
            # Calculate source reliability
            source_reliability = self.SOURCE_RELIABILITY.get(
                source_id, 
                self.SOURCE_RELIABILITY.get(source_name, self.SOURCE_RELIABILITY['default'])
            )
            
            # Extract symbols from content
            symbols = self._extract_symbols_from_text(article, target_symbols)
            
            # Analyze sentiment (basic keyword-based)
            sentiment = self._analyze_basic_sentiment(title, description)
            
            # Categorize the article
            categories = self._categorize_article(title, description)
            
            # Generate relevance scores
            relevance_scores = self._calculate_relevance_scores(article, symbols, target_symbols)
            
            return UnifiedNewsItem(
                id=f"newsapi_{self._generate_id(article)}",
                title=title,
                summary=description,
                content=article.get('content'),
                url=url,
                source=f"newsapi_{source_id}",
                source_reliability=source_reliability,
                published_at=published_at,
                symbols=symbols,
                sentiment=sentiment,
                relevance_scores=relevance_scores,
                categories=categories,
                language='en',  # NewsAPI filters by language
                metadata={
                    'author': article.get('author'),
                    'urlToImage': article.get('urlToImage'),
                    'source_name': source_name,
                    'source_id': source_id
                }
            )
            
        except Exception as e:
            logger.error(f"Error converting NewsAPI article: {e}")
            return None
    
    def _extract_symbols_from_text(
        self, 
        article: Dict[str, Any], 
        target_symbols: List[str] = None
    ) -> List[str]:
        """Extract stock symbols from article text"""
        symbols = set()
        
        # Combine all text
        text = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}"
        text_upper = text.upper()
        
        # Look for explicit symbol patterns
        symbol_patterns = [
            r'\$([A-Z]{1,5})\b',  # $AAPL
            r'\b([A-Z]{1,5})\s+stock\b',  # AAPL stock  
            r'\b([A-Z]{1,5})\s+shares\b',  # AAPL shares
            r'\(([A-Z]{1,5})\)',  # (AAPL)
            r'\b([A-Z]{2,5})\s+ticker\b'  # AAPL ticker
        ]
        
        for pattern in symbol_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1]
                if len(match) <= 5 and match.isalpha():
                    symbols.add(match.upper())
        
        # Check for target symbols mentioned in text
        if target_symbols:
            for symbol in target_symbols:
                if symbol.upper() in text_upper:
                    symbols.add(symbol.upper())
        
        return list(symbols)
    
    def _analyze_basic_sentiment(self, title: str, description: str) -> Optional[float]:
        """Basic sentiment analysis using keyword matching"""
        positive_keywords = [
            'growth', 'increase', 'rise', 'surge', 'boom', 'profit', 'gain',
            'success', 'bullish', 'upgrade', 'beat', 'exceeded', 'strong',
            'positive', 'optimistic', 'rally', 'breakthrough', 'expansion'
        ]
        
        negative_keywords = [
            'decline', 'fall', 'drop', 'crash', 'loss', 'cut', 'reduce',
            'bearish', 'downgrade', 'miss', 'weak', 'concern', 'risk',
            'negative', 'pessimistic', 'recession', 'bankruptcy', 'lawsuit'
        ]
        
        text = f"{title} {description}".lower()
        
        positive_count = sum(1 for word in positive_keywords if word in text)
        negative_count = sum(1 for word in negative_keywords if word in text)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        # Return score between -1 and 1
        sentiment = (positive_count - negative_count) / total
        return max(-1.0, min(1.0, sentiment))
    
    def _categorize_article(self, title: str, description: str) -> List[str]:
        """Categorize article based on content"""
        text = f"{title} {description}".lower()
        categories = []
        
        category_keywords = {
            'earnings': ['earnings', 'revenue', 'profit', 'quarter', 'eps'],
            'merger': ['merger', 'acquisition', 'buyout', 'takeover'],
            'ipo': ['ipo', 'initial public offering', 'public offering'],
            'regulatory': ['sec', 'regulation', 'compliance', 'filing', 'regulatory'],
            'economic': ['fed', 'federal reserve', 'interest rate', 'inflation', 'gdp'],
            'technology': ['tech', 'software', 'ai', 'artificial intelligence', 'digital'],
            'energy': ['oil', 'gas', 'energy', 'renewable', 'solar', 'wind'],
            'healthcare': ['health', 'pharma', 'medical', 'drug', 'biotech'],
            'financial': ['bank', 'credit', 'loan', 'financial', 'fintech']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in text for keyword in keywords):
                categories.append(category)
        
        return categories if categories else ['general']
    
    def _calculate_relevance_scores(
        self, 
        article: Dict[str, Any], 
        extracted_symbols: List[str],
        target_symbols: List[str] = None
    ) -> Dict[str, float]:
        """Calculate relevance scores for symbols"""
        scores = {}
        
        if not target_symbols:
            return scores
        
        text = f"{article.get('title', '')} {article.get('description', '')}"
        text_lower = text.lower()
        
        for symbol in target_symbols:
            score = 0.0
            symbol_lower = symbol.lower()
            
            # Direct mentions
            if symbol_lower in text_lower:
                score += 0.8
            
            # Symbol in extracted symbols
            if symbol.upper() in [s.upper() for s in extracted_symbols]:
                score += 0.5
            
            # Title mentions (more important)
            if symbol_lower in article.get('title', '').lower():
                score += 0.3
            
            # Contextual mentions
            contexts = [f'{symbol_lower} stock', f'{symbol_lower} shares']
            for context in contexts:
                if context in text_lower:
                    score += 0.2
            
            if score > 0:
                scores[symbol.upper()] = min(1.0, score)
        
        return scores
    
    def _generate_id(self, article: Dict[str, Any]) -> str:
        """Generate unique ID for article"""
        url = article.get('url', '')
        if url:
            return hashlib.md5(url.encode()).hexdigest()[:16]
        
        # Fallback to title + source hash
        title = article.get('title', '')
        source = article.get('source', {}).get('id', '')
        return hashlib.md5(f"{title}{source}".encode()).hexdigest()[:16]
    
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
            
            # Calculate confidence
            sentiment_variance = sum((s - avg_sentiment) ** 2 for s in sentiments) / len(sentiments)
            confidence = max(0.0, 1.0 - (sentiment_variance ** 0.5))
            
            return {
                'symbol': symbol,
                'sentiment_score': avg_sentiment,
                'sentiment_label': sentiment_label,
                'confidence': confidence,
                'article_count': len(news_items),
                'sources': list(set(item.source for item in news_items)),
                'time_range_hours': lookback_hours
            }
            
        except Exception as e:
            logger.error(f"Error getting NewsAPI sentiment for {symbol}: {e}")
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
        requests_remaining = self.current_limit['requests_per_day'] - self.requests_made_today
        
        return {
            'plan': self.plan,
            'requests_per_day': self.current_limit['requests_per_day'],
            'requests_remaining': max(0, requests_remaining),
            'requests_made_today': self.requests_made_today,
            'results_per_request': self.current_limit['results_per_request'],
            'last_request_time': self.last_request_time.isoformat() if self.last_request_time else None
        }
    
    async def health_check(self) -> bool:
        """Check if NewsAPI is accessible"""
        try:
            # Make a simple request to check API status
            params = {
                'q': 'test',
                'pageSize': 1,
                'language': 'en'
            }
            
            data = await self._make_request('everything', params)
            return data is not None and data.get('status') == 'ok'
            
        except Exception as e:
            logger.error(f"NewsAPI health check failed: {e}")
            return False
    
    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None