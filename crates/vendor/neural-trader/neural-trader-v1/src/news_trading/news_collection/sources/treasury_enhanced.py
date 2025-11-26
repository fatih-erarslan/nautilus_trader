"""Enhanced Treasury Direct source for bond market data - GREEN phase"""

import aiohttp
from datetime import datetime
from typing import List, AsyncIterator, Dict, Any, Optional
import logging

from news.models import NewsItem
from news_trading.news_collection.base import NewsSource

logger = logging.getLogger(__name__)


class TreasuryEnhancedSource(NewsSource):
    """Enhanced Treasury Direct source for bond auction results and yield data"""
    
    def __init__(self):
        super().__init__("treasury_direct")
        self.base_url = "https://api.treasurydirect.gov"
        
        # Bid-to-cover thresholds for demand assessment
        self.btc_thresholds = {
            "weak": 2.0,
            "moderate": 2.3,
            "strong": 2.5,
            "very_strong": 2.8
        }
        
    async def fetch_latest(self, limit: int = 100) -> List[NewsItem]:
        """Fetch latest Treasury auction results"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/auctions/recent"
            params = {"limit": limit}
            
            try:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    items = []
                    for auction in data.get("auctions", []):
                        news_item = self._parse_auction_result(auction)
                        if news_item:
                            items.append(news_item)
                    
                    return items
                    
            except Exception as e:
                logger.error(f"Error fetching Treasury auctions: {e}")
                return []
    
    def _parse_auction_result(self, auction: Dict[str, Any]) -> Optional[NewsItem]:
        """Parse Treasury auction result into NewsItem"""
        try:
            security_type = auction["security_type"]
            high_yield = float(auction["high_yield"])
            bid_to_cover = float(auction["bid_to_cover"])
            
            # Assess demand strength
            demand_strength = self._assess_demand_strength(bid_to_cover)
            
            # Create title
            title = f"{security_type} Auction: {high_yield}% yield, {bid_to_cover} bid-to-cover"
            
            # Build content
            content = (
                f"Treasury auction results for {security_type}:\n"
                f"High Yield: {high_yield}%\n"
                f"Bid-to-Cover Ratio: {bid_to_cover}\n"
                f"Indirect Bidders: {auction['indirect_bidders_pct']}%\n"
                f"Direct Bidders: {auction['direct_bidders_pct']}%\n"
                f"Primary Dealers: {auction['primary_dealers_pct']}%"
            )
            
            # Build metadata
            metadata = {
                "bond_type": security_type,
                "yield": high_yield,
                "bid_to_cover": bid_to_cover,
                "demand_strength": demand_strength,
                "indirect_bidders_pct": float(auction["indirect_bidders_pct"]),
                "direct_bidders_pct": float(auction["direct_bidders_pct"]),
                "primary_dealers_pct": float(auction["primary_dealers_pct"]),
                "auction_date": auction["auction_date"],
                "issue_date": auction["issue_date"],
                "maturity_date": auction["maturity_date"]
            }
            
            # Extract maturity for entity
            maturity = self._extract_maturity(security_type)
            entities = [f"{maturity}-Treasury"] if maturity else ["Treasury"]
            
            return NewsItem(
                id=f"treasury-{auction['id']}",
                title=title,
                content=content,
                source=self.source_name,
                timestamp=datetime.fromisoformat(auction["auction_date"]),
                url=f"https://treasurydirect.gov/auctions/{auction['id']}",
                entities=entities,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error parsing auction result: {e}")
            return None
    
    def _assess_demand_strength(self, bid_to_cover: float) -> str:
        """Assess demand strength based on bid-to-cover ratio"""
        if bid_to_cover >= self.btc_thresholds["very_strong"]:
            return "very_strong"
        elif bid_to_cover >= self.btc_thresholds["strong"]:
            return "strong"
        elif bid_to_cover >= self.btc_thresholds["moderate"]:
            return "moderate"
        else:
            return "weak"
    
    def _extract_maturity(self, security_type: str) -> Optional[str]:
        """Extract maturity period from security type"""
        import re
        
        # Match patterns like "10-Year Note", "2-Year Note", "30-Year Bond"
        match = re.search(r'(\d+)-Year', security_type)
        if match:
            return f"{match.group(1)}Y"
        
        # Match patterns like "13-Week Bill", "26-Week Bill"
        match = re.search(r'(\d+)-Week', security_type)
        if match:
            weeks = int(match.group(1))
            if weeks <= 13:
                return "3M"
            elif weeks <= 26:
                return "6M"
            else:
                return "1Y"
        
        return None
    
    def _analyze_tips_auction(self, auction: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze TIPS auction for inflation expectations"""
        real_yield = float(auction.get("real_yield", 0))
        breakeven_inflation = float(auction.get("breakeven_inflation", 0))
        
        # Determine market sentiment based on breakeven inflation
        if breakeven_inflation > 3.0:
            sentiment = "high_inflation_concern"
            opportunity = "inflation_hedge"
        elif breakeven_inflation > 2.5:
            sentiment = "moderate_inflation_concern"
            opportunity = "inflation_hedge"
        elif breakeven_inflation < 1.5:
            sentiment = "deflation_concern"
            opportunity = "nominal_bonds"
        else:
            sentiment = "neutral"
            opportunity = "balanced"
        
        return {
            "inflation_expectations": breakeven_inflation,
            "real_yield": real_yield,
            "market_sentiment": sentiment,
            "trading_opportunity": opportunity
        }
    
    async def stream(self) -> AsyncIterator[NewsItem]:
        """Stream Treasury auction results (not implemented)"""
        raise NotImplementedError("Treasury Direct API does not support streaming")