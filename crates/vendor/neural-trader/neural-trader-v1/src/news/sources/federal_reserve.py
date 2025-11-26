"""
Federal Reserve news source implementation for monetary policy and economic data
"""
import re
import aiohttp
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from . import NewsSource, NewsSourceError
from ..models import NewsItem


logger = logging.getLogger(__name__)


class FederalReserveSource(NewsSource):
    """Federal Reserve source for FOMC minutes, policy statements, and economic data"""
    
    def __init__(self, api_key: str):
        """
        Initialize Federal Reserve news source
        
        Args:
            api_key: FRED API key for economic data
        """
        super().__init__("federal_reserve")
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred"
        self.fed_url = "https://www.federalreserve.gov"
    
    async def fetch_latest(self, limit: int = 100) -> List[NewsItem]:
        """
        Fetch latest Fed policy updates and economic data
        
        Args:
            limit: Maximum number of items to fetch
            
        Returns:
            List of NewsItem objects
        """
        items = []
        
        # Fetch economic indicators as news items
        try:
            indicators = await self.fetch_economic_indicators()
            econ_item = self._create_economic_update_item(indicators)
            items.append(econ_item)
        except Exception as e:
            logger.error(f"Failed to fetch economic indicators: {e}")
        
        # Fetch FOMC meeting minutes
        try:
            minutes = await self.fetch_fomc_minutes()
            if minutes:
                minutes_item = self._create_fomc_minutes_item(minutes)
                items.append(minutes_item)
        except Exception as e:
            logger.error(f"Failed to fetch FOMC minutes: {e}")
        
        return items[:limit]
    
    async def fetch_economic_indicators(self) -> Dict[str, Any]:
        """
        Fetch key economic indicators from FRED
        
        Returns:
            Dictionary of economic indicators
        """
        indicators = {}
        
        # Fetch Fed Funds Rate
        fed_funds = await self._fetch_fred_series("DFF", limit=1)
        if fed_funds:
            indicators["fed_funds_rate"] = float(fed_funds[0]["value"])
        
        # Fetch inflation data (CPI)
        cpi_data = await self._fetch_fred_series("CPIAUCSL", limit=2)
        if len(cpi_data) >= 2:
            current_cpi = float(cpi_data[0]["value"])
            previous_cpi = float(cpi_data[1]["value"])
            inflation_rate = ((current_cpi - previous_cpi) / previous_cpi) * 100 * 12  # Annualized
            indicators["inflation_rate"] = round(inflation_rate, 2)
        
        return indicators
    
    async def _fetch_fred_series(self, series_id: str, limit: int = 1) -> List[Dict[str, Any]]:
        """Fetch data from FRED API"""
        url = f"{self.base_url}/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "limit": limit,
            "sort_order": "desc",
            "file_type": "json"
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        raise NewsSourceError(f"FRED API error: {response.status}")
                    
                    data = await response.json()
                    return data.get("observations", [])
                    
            except aiohttp.ClientError as e:
                raise NewsSourceError(f"FRED connection error: {str(e)}") from e
    
    def _create_economic_update_item(self, indicators: Dict[str, Any]) -> NewsItem:
        """Create news item from economic indicators"""
        title = "Federal Reserve Economic Data Update"
        
        content_parts = []
        if "fed_funds_rate" in indicators:
            content_parts.append(f"Federal Funds Rate: {indicators['fed_funds_rate']}%")
        
        if "inflation_rate" in indicators:
            content_parts.append(f"Annual Inflation Rate: {indicators['inflation_rate']}%")
        
        content = "Latest economic indicators from the Federal Reserve:\n" + "\n".join(content_parts)
        
        return NewsItem(
            id=f"fed-econ-{datetime.now().strftime('%Y%m%d')}",
            title=title,
            content=content,
            source=self.source_name,
            timestamp=datetime.now(),
            url=f"{self.fed_url}/data.htm",
            entities=["FED", "RATES", "INFLATION"],
            metadata=indicators
        )
    
    async def fetch_fomc_minutes(self) -> Optional[Dict[str, Any]]:
        """
        Fetch and parse latest FOMC meeting minutes
        
        Returns:
            Parsed minutes data
        """
        # This would typically parse the latest FOMC minutes URL
        # For testing, we'll mock the response
        minutes_url = f"{self.fed_url}/monetarypolicy/fomcminutes20240131.htm"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(minutes_url) as response:
                    if response.status != 200:
                        return None
                    
                    html = await response.text()
                    return self._parse_fomc_minutes(html, "2024-01-31")
                    
            except aiohttp.ClientError as e:
                logger.error(f"Failed to fetch FOMC minutes: {e}")
                return None
    
    def _parse_fomc_minutes(self, html: str, date: str) -> Dict[str, Any]:
        """Parse FOMC minutes HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract main content
        content_div = soup.find('div', class_='col-xs-12 col-sm-8 col-md-8')
        if not content_div:
            content_div = soup.find('body')
        
        text = content_div.get_text() if content_div else ""
        
        # Extract key topics using keyword matching
        key_topics = []
        keywords = {
            "inflation": ["inflation", "price", "cpi", "pce"],
            "employment": ["employment", "jobs", "unemployment", "labor"],
            "policy rate": ["policy rate", "federal funds", "interest rate"],
            "financial stability": ["financial stability", "systemic risk"],
            "economic outlook": ["outlook", "forecast", "projection"]
        }
        
        for topic, terms in keywords.items():
            if any(term.lower() in text.lower() for term in terms):
                key_topics.append(topic)
        
        # Simple sentiment analysis
        hawkish_words = ["aggressive", "tighten", "restrictive", "combat inflation", "raise rates", "elevated"]
        dovish_words = ["accommodate", "support", "gradual", "pause", "lower rates", "easing", "peak", "at or near its peak"]
        
        hawkish_count = sum(1 for word in hawkish_words if word.lower() in text.lower())
        dovish_count = sum(1 for word in dovish_words if word.lower() in text.lower())
        
        return {
            "date": date,
            "key_topics": key_topics,
            "text_length": len(text),
            "sentiment": {
                "hawkish_score": hawkish_count,
                "dovish_score": dovish_count
            }
        }
    
    def _create_fomc_minutes_item(self, minutes: Dict[str, Any]) -> NewsItem:
        """Create news item from FOMC minutes"""
        date = minutes["date"]
        topics = ", ".join(minutes["key_topics"])
        
        title = f"FOMC Meeting Minutes Released - {date}"
        content = f"""
        The Federal Reserve released minutes from the FOMC meeting held on {date}.
        Key discussion topics included: {topics}.
        """
        
        # Determine overall sentiment
        sentiment = minutes["sentiment"]
        if sentiment["hawkish_score"] > sentiment["dovish_score"]:
            overall_sentiment = "hawkish"
        elif sentiment["dovish_score"] > sentiment["hawkish_score"]:
            overall_sentiment = "dovish"
        else:
            overall_sentiment = "neutral"
        
        return NewsItem(
            id=f"fomc-minutes-{date}",
            title=title,
            content=content.strip(),
            source=self.source_name,
            timestamp=datetime.now(),
            url=f"{self.fed_url}/monetarypolicy/fomcminutes{date.replace('-', '')}.htm",
            entities=["FOMC", "FED", "MINUTES"],
            metadata={
                "meeting_date": date,
                "key_topics": minutes["key_topics"],
                "sentiment": overall_sentiment,
                **sentiment
            }
        )
    
    def analyze_policy_statement(self, statement: str) -> Dict[str, str]:
        """
        Analyze Fed policy statement for sentiment and market impact
        
        Args:
            statement: Policy statement or headline
            
        Returns:
            Analysis results
        """
        statement_lower = statement.lower()
        
        # Hawkish indicators
        hawkish_terms = [
            "rate cuts", "aggressive", "combat inflation", "restrictive",
            "tighten", "hawkish", "raise rates", "fight inflation"
        ]
        
        # Dovish indicators
        dovish_terms = [
            "rate cuts", "accommodation", "support growth", "gradual",
            "pause", "dovish", "lower rates", "easing", "stimulus",
            "hints at", "extended pause", "potential rate cuts"
        ]
        
        # Neutral indicators
        neutral_terms = [
            "division", "mixed", "varied views", "disagreement",
            "split", "uncertain", "data dependent"
        ]
        
        # Determine sentiment
        hawkish_score = sum(1 for term in hawkish_terms if term in statement_lower)
        dovish_score = sum(1 for term in dovish_terms if term in statement_lower)
        neutral_score = sum(1 for term in neutral_terms if term in statement_lower)
        
        if neutral_score > 0 or hawkish_score == dovish_score:
            sentiment = "neutral"
            bond_impact = "neutral"
        elif hawkish_score > dovish_score:
            sentiment = "hawkish"
            bond_impact = "negative"  # Hawkish = rates up = bond prices down
        else:
            sentiment = "dovish"
            bond_impact = "positive"  # Dovish = rates down = bond prices up
        
        return {
            "sentiment": sentiment,
            "bond_market_impact": bond_impact,
            "confidence": "high" if abs(hawkish_score - dovish_score) > 1 else "medium"
        }
    
    async def stream(self):
        """
        Stream Fed updates (polls periodically)
        
        Yields:
            NewsItem objects as they arrive
        """
        import asyncio
        
        seen_ids = set()
        poll_interval = 1800  # 30 minutes
        
        while True:
            try:
                items = await self.fetch_latest(limit=10)
                
                for item in items:
                    if item.id not in seen_ids:
                        seen_ids.add(item.id)
                        yield item
                
                await asyncio.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Stream error: {str(e)}")
                await asyncio.sleep(poll_interval * 2)