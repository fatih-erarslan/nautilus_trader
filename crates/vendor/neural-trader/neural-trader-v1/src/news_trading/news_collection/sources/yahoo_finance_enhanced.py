"""Enhanced Yahoo Finance news source with earnings and momentum detection - GREEN phase"""

import aiohttp
import re
from datetime import datetime
from typing import List, AsyncIterator, Dict, Any, Optional
import logging

from news.models import NewsItem
from news_trading.news_collection.base import NewsSource

logger = logging.getLogger(__name__)


class YahooFinanceEnhancedSource(NewsSource):
    """Enhanced Yahoo Finance news source with trading signal detection"""
    
    def __init__(self):
        super().__init__("yahoo_finance")
        self.base_url = "https://query1.finance.yahoo.com"
        self.ticker_patterns = self._compile_ticker_patterns()
        
    def _compile_ticker_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for common tickers"""
        tickers = {
            "AAPL": r"\b(Apple|AAPL)\b",
            "GOOGL": r"\b(Google|Alphabet|GOOGL|GOOG)\b",
            "MSFT": r"\b(Microsoft|MSFT)\b",
            "AMZN": r"\b(Amazon|AMZN)\b",
            "TSLA": r"\b(Tesla|TSLA)\b",
            "NVDA": r"\b(NVIDIA|Nvidia|NVDA)\b",
            "META": r"\b(Meta|Facebook|META|FB)\b",
            "BRK": r"\b(Berkshire|BRK\.A|BRK\.B)\b",
        }
        return {ticker: re.compile(pattern, re.IGNORECASE) for ticker, pattern in tickers.items()}
    
    async def fetch_latest(self, limit: int = 100) -> List[NewsItem]:
        """Fetch latest news from Yahoo Finance"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/v1/finance/news"
            params = {
                "category": "generalnews",
                "count": limit
            }
            
            try:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    items = []
                    for article in data.get("news", []):
                        news_item = self._parse_article(article)
                        if news_item:
                            items.append(news_item)
                    
                    return items
                    
            except Exception as e:
                logger.error(f"Error fetching from Yahoo Finance: {e}")
                return []
    
    def _parse_article(self, article: Dict[str, Any]) -> Optional[NewsItem]:
        """Parse Yahoo Finance article into NewsItem"""
        try:
            # Extract entities
            entities = self._extract_entities(article.get("title", "") + " " + article.get("summary", ""))
            
            # Create NewsItem
            news_item = NewsItem(
                id=f"yahoo-{article['uuid']}",
                title=article["title"],
                content=article.get("summary", ""),
                source=self.source_name,
                timestamp=datetime.fromtimestamp(article["providerPublishTime"]),
                url=article["link"],
                entities=entities,
                metadata=self._extract_metadata(article)
            )
            
            return news_item
            
        except Exception as e:
            logger.error(f"Error parsing article: {e}")
            return None
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract stock ticker entities from text"""
        entities = []
        
        for ticker, pattern in self.ticker_patterns.items():
            if pattern.search(text):
                entities.append(ticker)
        
        # Also look for explicit ticker mentions
        ticker_match = re.findall(r'\b[A-Z]{2,5}\b', text)
        for ticker in ticker_match:
            if ticker not in entities and len(ticker) <= 5:
                entities.append(ticker)
        
        return list(set(entities))
    
    def _extract_metadata(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata including trading signals"""
        metadata = {
            "publisher": article.get("publisher", ""),
            "provider_publish_time": article.get("providerPublishTime", 0)
        }
        
        # Check for earnings-related content
        title_lower = article.get("title", "").lower()
        summary_lower = article.get("summary", "").lower()
        full_text = title_lower + " " + summary_lower
        
        # Earnings detection
        if any(term in full_text for term in ["earnings", "eps", "revenue", "quarter", "q1", "q2", "q3", "q4"]):
            metadata.update(self._extract_earnings_metadata(article))
        
        return metadata
    
    def _extract_earnings_metadata(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Extract earnings-specific metadata for momentum trading"""
        metadata = {"is_earnings": True}
        
        full_text = article.get("title", "") + " " + article.get("summary", "")
        
        # Check for earnings beat
        beat_patterns = [
            r"beat(?:s)?\s+(?:by\s+)?\$?(\d+\.?\d*)",
            r"exceed(?:s|ed)?\s+expectations",
            r"surpass(?:es|ed)?\s+estimates",
            r"top(?:s|ped)?\s+expectations"
        ]
        
        for pattern in beat_patterns:
            if re.search(pattern, full_text, re.IGNORECASE):
                metadata["earnings_beat"] = True
                
                # Extract EPS surprise if available
                eps_match = re.search(r"EPS.*?(?:beat|exceed|surpass).*?\$?(\d+\.?\d*)", full_text, re.IGNORECASE)
                if eps_match:
                    metadata["eps_surprise"] = float(eps_match.group(1))
                else:
                    metadata["eps_surprise"] = 0.15  # Default if beat but no number
                
                break
        
        # Extract revenue growth
        revenue_match = re.search(r"revenue.*?(?:up|grow|increase|rise).*?(\d+)%", full_text, re.IGNORECASE)
        if revenue_match:
            metadata["revenue_growth"] = float(revenue_match.group(1)) / 100
        
        # Determine momentum signal
        if metadata.get("earnings_beat"):
            if metadata.get("revenue_growth", 0) > 0.2:
                metadata["momentum_signal"] = "strong_positive"
            else:
                metadata["momentum_signal"] = "positive"
        else:
            metadata["momentum_signal"] = "neutral"
        
        return metadata
    
    async def stream(self) -> AsyncIterator[NewsItem]:
        """Stream news items in real-time (not implemented for Yahoo Finance)"""
        raise NotImplementedError("Yahoo Finance does not support streaming")