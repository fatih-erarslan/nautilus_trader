"""Technical analysis news source for swing trading opportunities - GREEN phase"""

import aiohttp
from datetime import datetime
from typing import List, AsyncIterator, Dict, Any, Optional
import logging

from news.models import NewsItem
from news_trading.news_collection.base import NewsSource

logger = logging.getLogger(__name__)


class TechnicalNewsSource(NewsSource):
    """Technical analysis news source for detecting breakouts and swing trading opportunities"""
    
    def __init__(self):
        super().__init__("technical_news")
        self.base_url = "https://api.technical-analysis.com"
        
        # Technical indicators that suggest swing trading opportunities
        self.swing_indicators = {
            "200_MA_breakout": {"strength": 0.8, "holding_days": "5-15"},
            "50_MA_breakout": {"strength": 0.7, "holding_days": "3-10"},
            "volume_spike": {"strength": 0.6, "holding_days": "2-5"},
            "bullish_flag": {"strength": 0.75, "holding_days": "3-10"},
            "cup_and_handle": {"strength": 0.85, "holding_days": "10-30"},
            "ascending_triangle": {"strength": 0.8, "holding_days": "5-20"},
            "rsi_oversold_bounce": {"strength": 0.7, "holding_days": "3-10"},
            "macd_bullish_cross": {"strength": 0.65, "holding_days": "5-15"}
        }
        
    async def fetch_latest(self, limit: int = 100) -> List[NewsItem]:
        """Fetch latest technical breakout news"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/breakouts"
            params = {"limit": limit}
            
            try:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    items = []
                    for article in data.get("articles", []):
                        news_item = self._parse_technical_article(article)
                        if news_item:
                            items.append(news_item)
                    
                    return items
                    
            except Exception as e:
                logger.error(f"Error fetching technical news: {e}")
                return []
    
    def _parse_technical_article(self, article: Dict[str, Any]) -> Optional[NewsItem]:
        """Parse technical analysis article into NewsItem"""
        try:
            ticker = article.get("ticker", "")
            indicators = article.get("indicators", [])
            
            # Calculate signal strength based on indicators
            signal_strength = self._calculate_signal_strength(indicators)
            
            # Determine holding period
            holding_period = self._determine_holding_period(indicators)
            
            # Build metadata
            metadata = {
                "swing_trade_signal": signal_strength > 0.6,
                "technical_indicators": indicators,
                "signal_strength": signal_strength,
                "suggested_holding_period": holding_period,
                "ticker": ticker,
                "technical_score": article.get("technical_score", signal_strength)
            }
            
            # Add price levels if available
            if "support_level" in article:
                metadata["support_level"] = article["support_level"]
            if "resistance_level" in article:
                metadata["resistance_level"] = article["resistance_level"]
            
            return NewsItem(
                id=f"tech-{article['id']}",
                title=article["headline"],
                content=article.get("content", ""),
                source=self.source_name,
                timestamp=datetime.fromisoformat(article["timestamp"].replace('Z', '+00:00')),
                url=article.get("url", ""),
                entities=[ticker] if ticker else [],
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error parsing technical article: {e}")
            return None
    
    def _calculate_signal_strength(self, indicators: List[str]) -> float:
        """Calculate combined signal strength from multiple indicators"""
        if not indicators:
            return 0.0
            
        total_strength = 0.0
        count = 0
        
        for indicator in indicators:
            if indicator in self.swing_indicators:
                total_strength += self.swing_indicators[indicator]["strength"]
                count += 1
        
        # Average strength with bonus for multiple confirming indicators
        if count > 0:
            avg_strength = total_strength / count
            # Add bonus for multiple indicators (max 20% bonus)
            confirmation_bonus = min(0.2, (count - 1) * 0.05)
            return min(1.0, avg_strength + confirmation_bonus)
        
        return 0.5  # Default if unknown indicators
    
    def _determine_holding_period(self, indicators: List[str]) -> str:
        """Determine suggested holding period based on indicators"""
        periods = []
        
        for indicator in indicators:
            if indicator in self.swing_indicators:
                periods.append(self.swing_indicators[indicator]["holding_days"])
        
        if not periods:
            return "3-10 days"  # Default swing trading period
        
        # Return the most common period or the first one
        return periods[0]
    
    async def stream(self) -> AsyncIterator[NewsItem]:
        """Stream technical breakouts in real-time (not implemented)"""
        raise NotImplementedError("Technical news API does not support streaming")