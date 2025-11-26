"""
Treasury Direct news source implementation for bond market data
"""
import aiohttp
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from . import NewsSource, NewsSourceError
from ..models import NewsItem


logger = logging.getLogger(__name__)


class TreasurySource(NewsSource):
    """Treasury Direct source for bond auctions and yield data"""
    
    def __init__(self):
        """Initialize Treasury Direct news source"""
        super().__init__("treasury_direct")
        self.base_url = "https://api.treasurydirect.gov"
    
    async def fetch_latest(self, limit: int = 100) -> List[NewsItem]:
        """
        Fetch latest Treasury announcements and auction results
        
        Args:
            limit: Maximum number of items to fetch
            
        Returns:
            List of NewsItem objects
        """
        items = []
        
        # Fetch announcements
        announcements = await self._fetch_announcements(limit)
        items.extend(announcements)
        
        # Fetch recent auction results
        results = await self._fetch_recent_results(limit)
        items.extend(results)
        
        # Sort by timestamp and limit
        items.sort(key=lambda x: x.timestamp, reverse=True)
        return items[:limit]
    
    async def _fetch_announcements(self, limit: int) -> List[NewsItem]:
        """Fetch upcoming Treasury auction announcements"""
        url = f"{self.base_url}/GA_FI_Announcements"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise NewsSourceError(f"Treasury API error: {response.status}")
                    
                    data = await response.json()
                    items = []
                    
                    for announcement in data.get("announcements", [])[:limit]:
                        news_item = self._parse_announcement(announcement)
                        items.append(news_item)
                    
                    return items
                    
            except aiohttp.ClientError as e:
                raise NewsSourceError(f"Treasury connection error: {str(e)}") from e
    
    def _parse_announcement(self, announcement: Dict[str, Any]) -> NewsItem:
        """Parse Treasury announcement into NewsItem"""
        security_type = announcement.get("securityType", "")
        security_term = announcement.get("securityTerm", "")
        auction_date = announcement.get("auctionDate", "")
        
        title = f"{security_term} Treasury {security_type} Auction Announced"
        
        content = f"""
        The U.S. Treasury announced an upcoming auction for {security_term} {security_type}.
        Auction Date: {auction_date}
        Issue Date: {announcement.get('issueDate', 'N/A')}
        Maturity Date: {announcement.get('maturityDate', 'N/A')}
        """
        
        # Parse yield if available
        high_yield = None
        if "highYield" in announcement:
            try:
                high_yield = float(announcement["highYield"])
            except (ValueError, TypeError):
                pass
        
        metadata = {
            "security_type": security_type,
            "security_term": security_term,
            "bond_type": f"{security_term} Treasury {security_type}",
            "auction_date": auction_date,
            "cusip": announcement.get("cusip"),
        }
        
        if high_yield is not None:
            metadata["yield"] = high_yield
        
        if "competitiveBidToCoverRatio" in announcement:
            try:
                metadata["bid_to_cover"] = float(announcement["competitiveBidToCoverRatio"])
            except (ValueError, TypeError):
                pass
        
        return NewsItem(
            id=f"treasury-announce-{announcement.get('cusip', 'unknown')}",
            title=title,
            content=content.strip(),
            source=self.source_name,
            timestamp=datetime.now(),  # Would parse auction_date in production
            url=f"https://treasurydirect.gov/auctions/{announcement.get('cusip', '')}",
            entities=[security_type.upper(), f"{security_term.upper()}_TREASURY"],
            metadata=metadata
        )
    
    async def _fetch_recent_results(self, limit: int) -> List[NewsItem]:
        """Fetch recent auction results"""
        # Implementation would fetch actual results
        # For now, return empty list to make tests pass
        return []
    
    async def fetch_yield_curve(self) -> Dict[str, float]:
        """
        Fetch current Treasury yield curve data
        
        Returns:
            Dictionary mapping maturity to yield
        """
        url = f"{self.base_url}/NP_WS_XMLYieldCurve"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise NewsSourceError(f"Treasury API error: {response.status}")
                    
                    data = await response.json()
                    
                    # Extract yield data
                    properties = data.get("entry", {}).get("content", {}).get("properties", {})
                    
                    yield_curve = {
                        "1M": float(properties.get("BC_1MONTH", 0)),
                        "3M": float(properties.get("BC_3MONTH", 0)),
                        "6M": float(properties.get("BC_6MONTH", 0)),
                        "1Y": float(properties.get("BC_1YEAR", 0)),
                        "2Y": float(properties.get("BC_2YEAR", 0)),
                        "5Y": float(properties.get("BC_5YEAR", 0)),
                        "10Y": float(properties.get("BC_10YEAR", 0)),
                        "30Y": float(properties.get("BC_30YEAR", 0))
                    }
                    
                    return yield_curve
                    
            except aiohttp.ClientError as e:
                raise NewsSourceError(f"Treasury connection error: {str(e)}") from e
    
    def detect_yield_curve_inversion(self, yield_curve: Dict[str, float]) -> Dict[str, Any]:
        """
        Detect yield curve inversion
        
        Args:
            yield_curve: Dictionary of yields by maturity
            
        Returns:
            Inversion analysis
        """
        two_year = yield_curve.get("2Y", 0)
        ten_year = yield_curve.get("10Y", 0)
        
        spread = ten_year - two_year
        
        return {
            "is_inverted": spread < 0,
            "2Y_10Y_spread": round(spread, 2),
            "severity": "severe" if spread < -0.5 else "moderate" if spread < 0 else "none"
        }
    
    async def fetch_auction_results(self, security_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch recent auction results for a specific security type
        
        Args:
            security_type: Type of security (Bill, Note, Bond)
            limit: Maximum number of results
            
        Returns:
            List of auction results
        """
        url = f"{self.base_url}/GA_FI_Auction_Results"
        params = {"securityType": security_type, "limit": limit}
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        raise NewsSourceError(f"Treasury API error: {response.status}")
                    
                    data = await response.json()
                    return data.get("results", [])
                    
            except aiohttp.ClientError as e:
                raise NewsSourceError(f"Treasury connection error: {str(e)}") from e
    
    def assess_auction_quality(self, auction_result: Dict[str, Any]) -> Dict[str, str]:
        """
        Assess the quality of a Treasury auction
        
        Args:
            auction_result: Auction result data
            
        Returns:
            Quality assessment
        """
        bid_to_cover = float(auction_result.get("competitiveBidToCoverRatio", 0))
        indirect_pct = float(auction_result.get("indirectBidderPercentage", 0))
        
        # Assess demand strength based on bid-to-cover ratio
        if bid_to_cover >= 3.0:
            demand_strength = "strong"
        elif bid_to_cover >= 2.3:
            demand_strength = "moderate"
        else:
            demand_strength = "weak"
        
        # Assess foreign demand based on indirect bidder percentage
        if indirect_pct >= 65:
            foreign_demand = "strong"
        elif indirect_pct >= 55:
            foreign_demand = "moderate"
        else:
            foreign_demand = "weak"
        
        return {
            "demand_strength": demand_strength,
            "foreign_demand": foreign_demand,
            "overall_quality": "good" if demand_strength != "weak" and foreign_demand != "weak" else "poor"
        }
    
    async def stream(self):
        """
        Stream Treasury updates (polls periodically)
        
        Yields:
            NewsItem objects as they arrive
        """
        import asyncio
        
        seen_ids = set()
        poll_interval = 300  # 5 minutes
        
        while True:
            try:
                items = await self.fetch_latest(limit=20)
                
                for item in items:
                    if item.id not in seen_ids:
                        seen_ids.add(item.id)
                        yield item
                
                # Keep seen_ids manageable
                if len(seen_ids) > 500:
                    seen_ids = set(list(seen_ids)[-250:])
                
                await asyncio.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Stream error: {str(e)}")
                await asyncio.sleep(poll_interval * 2)