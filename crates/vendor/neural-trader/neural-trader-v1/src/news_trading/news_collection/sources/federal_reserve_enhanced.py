"""Enhanced Federal Reserve news source - GREEN phase"""

import aiohttp
from datetime import datetime
from typing import List, AsyncIterator, Dict, Any, Optional
import logging

from news.models import NewsItem
from news_trading.news_collection.base import NewsSource

logger = logging.getLogger(__name__)


class FederalReserveEnhancedSource(NewsSource):
    """Enhanced Federal Reserve source for monetary policy announcements"""
    
    def __init__(self):
        super().__init__("federal_reserve")
        self.base_url = "https://api.federalreserve.gov"
        
        # Market impact assessment based on rate decisions
        self.rate_impact_map = {
            "increased": "hawkish",
            "decreased": "dovish",
            "unchanged": "neutral"
        }
        
    async def fetch_latest(self, limit: int = 100) -> List[NewsItem]:
        """Fetch latest Federal Reserve announcements"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/announcements/recent"
            params = {"limit": limit}
            
            try:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    items = []
                    for announcement in data.get("announcements", []):
                        news_item = self._parse_announcement(announcement)
                        if news_item:
                            items.append(news_item)
                    
                    return items
                    
            except Exception as e:
                logger.error(f"Error fetching Fed announcements: {e}")
                return []
    
    def _parse_announcement(self, announcement: Dict[str, Any]) -> Optional[NewsItem]:
        """Parse Federal Reserve announcement into NewsItem"""
        try:
            title = announcement["title"]
            content = announcement["content"]
            announcement_type = announcement.get("type", "general")
            
            # Extract key metrics for monetary policy announcements
            metadata = {
                "announcement_type": announcement_type
            }
            
            if announcement_type == "monetary_policy":
                key_metrics = announcement.get("key_metrics", {})
                metadata.update({
                    "fed_funds_rate": key_metrics.get("fed_funds_rate"),
                    "rate_decision": key_metrics.get("decision", "unchanged"),
                    "vote": key_metrics.get("vote", ""),
                    "market_impact": self._assess_market_impact(key_metrics.get("decision", "unchanged"))
                })
            
            # Extract entities
            entities = self._extract_fed_entities(content)
            
            return NewsItem(
                id=f"fed-{announcement['id']}",
                title=title,
                content=content,
                source=self.source_name,
                timestamp=datetime.fromisoformat(announcement["release_date"].replace('Z', '+00:00')),
                url=f"https://federalreserve.gov/announcements/{announcement['id']}",
                entities=entities,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error parsing Fed announcement: {e}")
            return None
    
    def _assess_market_impact(self, decision: str) -> str:
        """Assess market impact of Fed decision"""
        return self.rate_impact_map.get(decision, "neutral")
    
    def _extract_fed_entities(self, content: str) -> List[str]:
        """Extract relevant entities from Fed announcements"""
        entities = []
        
        # Common Fed-related entities
        fed_keywords = {
            "FOMC": "Federal Open Market Committee",
            "Fed": "Federal Reserve",
            "QE": "Quantitative Easing",
            "QT": "Quantitative Tightening"
        }
        
        for keyword, full_name in fed_keywords.items():
            if keyword in content or full_name in content:
                entities.append(keyword)
        
        # Also check for economic indicators mentioned
        indicators = ["GDP", "CPI", "PCE", "unemployment", "inflation"]
        for indicator in indicators:
            if indicator.lower() in content.lower():
                entities.append(indicator.upper())
        
        return list(set(entities))
    
    async def stream(self) -> AsyncIterator[NewsItem]:
        """Stream Fed announcements (not implemented)"""
        raise NotImplementedError("Federal Reserve API does not support streaming")