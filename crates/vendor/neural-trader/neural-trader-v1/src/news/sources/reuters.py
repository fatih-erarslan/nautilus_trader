"""
Reuters news source implementation for stock market news
"""
import re
import aiohttp
import logging
from typing import List, AsyncIterator, Dict, Any
from datetime import datetime
from . import NewsSource, NewsSourceError
from ..models import NewsItem


logger = logging.getLogger(__name__)


class ReutersSource(NewsSource):
    """Reuters news source for financial and market news"""
    
    def __init__(self, api_key: str):
        """
        Initialize Reuters news source
        
        Args:
            api_key: Reuters API key for authentication
        """
        super().__init__("reuters")
        self.api_key = api_key
        self.base_url = "https://api.reuters.com/v1"
        self.ws_url = "wss://stream.reuters.com/v1/news"
    
    async def fetch_latest(self, limit: int = 100) -> List[NewsItem]:
        """
        Fetch latest news articles from Reuters
        
        Args:
            limit: Maximum number of articles to fetch
            
        Returns:
            List of NewsItem objects
        """
        url = f"{self.base_url}/articles/latest"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }
        params = {"limit": limit}
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 500:
                        raise NewsSourceError(f"Reuters API error: Server error {response.status}")
                    elif response.status != 200:
                        raise NewsSourceError(f"Reuters API error: {response.status}")
                    
                    data = await response.json()
                    
                    return [
                        self._parse_article(article)
                        for article in data.get("articles", [])
                    ]
                    
            except aiohttp.ClientError as e:
                raise NewsSourceError(f"Reuters connection error: {str(e)}") from e
    
    def _parse_article(self, article: Dict[str, Any]) -> NewsItem:
        """Parse Reuters article data into NewsItem"""
        return NewsItem(
            id=f"reuters-{article['id']}",
            title=article['headline'],
            content=article['body'],
            source=self.source_name,
            timestamp=datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
            url=article['url'],
            entities=self._extract_entities(article['body']),
            metadata={
                "author": article.get('author'),
                "categories": article.get('categories', [])
            }
        )
    
    def _extract_entities(self, content: str) -> List[str]:
        """
        Extract stock symbols and entities from content
        
        Args:
            content: Article content
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Extract stock symbols in parentheses (e.g., (AAPL), (MSFT))
        ticker_pattern = r'\(([A-Z]{1,5})\)'
        tickers = re.findall(ticker_pattern, content)
        entities.extend(tickers)
        
        # Extract standalone tickers (e.g., TSLA, AAPL mentioned without parentheses)
        standalone_ticker_pattern = r'\b([A-Z]{2,5})\b(?![a-z])(?!\s*\))'
        standalone_tickers = re.findall(standalone_ticker_pattern, content)
        # Filter to likely stock symbols
        for ticker in standalone_tickers:
            if ticker not in entities and ticker not in ['THE', 'AND', 'FOR', 'INC', 'LLC', 'ETF']:
                entities.append(ticker)
        
        # Extract crypto symbols (e.g., Bitcoin (BTC))
        crypto_pattern = r'\b(?:Bitcoin|Ethereum|BTC|ETH)\b'
        cryptos = re.findall(crypto_pattern, content, re.IGNORECASE)
        entities.extend([c.upper() for c in cryptos if c.upper() not in entities])
        
        # Extract major indices
        index_patterns = [
            r'S&P 500',
            r'Nasdaq(?:\s+Composite)?',
            r'Dow Jones',
            r'Russell \d+',
            r'FTSE \d+',
            r'Nikkei'
        ]
        for pattern in index_patterns:
            indices = re.findall(pattern, content, re.IGNORECASE)
            for idx in indices:
                # Normalize Nasdaq variations
                if 'Nasdaq' in idx:
                    entities.append('Nasdaq')
                elif idx not in entities:
                    entities.append(idx)
        
        # Extract company names followed by ticker symbols
        company_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Inc\.|Corp\.|Company|Ltd\.)?\s*\(([A-Z]{1,5})\)'
        companies = re.findall(company_pattern, content)
        for company, ticker in companies:
            if ticker not in entities:
                entities.append(ticker)
        
        return entities
    
    async def stream(self) -> AsyncIterator[NewsItem]:
        """
        Stream real-time news from Reuters WebSocket
        
        Yields:
            NewsItem objects as they arrive
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.ws_connect(self.ws_url, headers=headers) as ws:
                    logger.info("Connected to Reuters WebSocket stream")
                    
                    while True:
                        msg = await ws.receive()
                        
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            import json
                            data = json.loads(msg.data)
                            
                            if data.get("type") == "article":
                                article = data.get("article", {})
                                yield self._parse_article(article)
                                
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket error: {ws.exception()}")
                            break
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            logger.info("WebSocket connection closed")
                            break
                            
            except Exception as e:
                logger.error(f"Stream error: {str(e)}")
                raise NewsSourceError(f"Reuters stream error: {str(e)}") from e