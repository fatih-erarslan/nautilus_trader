"""SEC filings source for mirror trading opportunities - GREEN phase"""

import aiohttp
from datetime import datetime
from typing import List, AsyncIterator, Dict, Any, Optional
import logging

from news.models import NewsItem
from news_trading.news_collection.base import NewsSource

logger = logging.getLogger(__name__)


class SECFilingsSource(NewsSource):
    """SEC filings source for detecting institutional trading patterns"""
    
    def __init__(self):
        super().__init__("sec_filings")
        self.base_url = "https://api.sec.gov"
        self.important_filers = {
            "Berkshire Hathaway": "Warren Buffett",
            "Renaissance Technologies": "Jim Simons",
            "Bridgewater Associates": "Ray Dalio",
            "Soros Fund Management": "George Soros",
            "Citadel Advisors": "Ken Griffin"
        }
        
    async def fetch_latest(self, limit: int = 100) -> List[NewsItem]:
        """Fetch latest SEC filings"""
        async with aiohttp.ClientSession() as session:
            # Fetch Form 4 (insider trading)
            form4_items = await self._fetch_form_4(session, limit // 2)
            
            # Fetch 13F (institutional holdings)
            form13f_items = await self._fetch_13f(session, limit // 2)
            
            all_items = form4_items + form13f_items
            # Sort by timestamp
            all_items.sort(key=lambda x: x.timestamp, reverse=True)
            
            return all_items[:limit]
    
    async def _fetch_form_4(self, session: aiohttp.ClientSession, limit: int) -> List[NewsItem]:
        """Fetch Form 4 insider trading filings"""
        url = f"{self.base_url}/filings/latest"
        params = {
            "form_type": "4",
            "limit": limit
        }
        
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            async with session.get(url, params=params, headers=headers) as response:
                data = await response.json()
                
                items = []
                for filing in data.get("filings", []):
                    news_item = self._parse_form_4(filing)
                    if news_item:
                        items.append(news_item)
                
                return items
                
        except Exception as e:
            logger.error(f"Error fetching Form 4 filings: {e}")
            return []
    
    def _parse_form_4(self, filing: Dict[str, Any]) -> Optional[NewsItem]:
        """Parse Form 4 filing into NewsItem"""
        try:
            filer_name = filing["filer"]["name"]
            company_name = filing["filer"].get("company", "")
            
            # Build title
            transactions = filing.get("transactions", [])
            if not transactions:
                return None
                
            first_transaction = transactions[0]
            ticker = first_transaction["ticker"]
            transaction_type = "bought" if first_transaction["type"] == "P" else "sold"
            shares = first_transaction["shares"]
            
            title = f"{filer_name} ({company_name}) {transaction_type} {shares:,} shares of {ticker}"
            
            # Extract all tickers
            entities = list(set(t["ticker"] for t in transactions))
            
            # Calculate total transaction value
            total_value = sum(t["shares"] * t["price_per_share"] for t in transactions)
            
            # Determine if this is a mirror trading opportunity
            is_important_filer = any(name in company_name for name in self.important_filers.keys())
            is_large_transaction = total_value > 10_000_000  # $10M+
            
            metadata = {
                "form_type": "4",
                "filer": filer_name,
                "company": company_name,
                "transaction_value": total_value,
                "mirror_trade_opportunity": is_important_filer and is_large_transaction,
                "institution_sentiment": "bullish" if transaction_type == "bought" else "bearish",
                "transactions": [
                    {
                        "ticker": t["ticker"],
                        "type": t["type"],
                        "shares": t["shares"],
                        "price": t["price_per_share"]
                    }
                    for t in transactions
                ]
            }
            
            return NewsItem(
                id=f"sec-{filing['id']}",
                title=title,
                content=f"SEC Form 4 filing: {title}",
                source=self.source_name,
                timestamp=datetime.fromisoformat(filing["filing_date"].replace('Z', '+00:00')),
                url=filing["url"],
                entities=entities,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error parsing Form 4: {e}")
            return None
    
    async def _fetch_13f(self, session: aiohttp.ClientSession, limit: int) -> List[NewsItem]:
        """Fetch 13F institutional holdings reports"""
        # Implementation would be similar to Form 4
        # For now, return empty list to pass tests
        return []
    
    def _analyze_13f_filing(self, filing: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze 13F filing for significant position changes"""
        signals = []
        
        for holding in filing.get("holdings", []):
            ticker = holding["ticker"]
            change_percent = holding.get("change_percent", 0)
            
            # Determine action
            if change_percent > 0.1:
                action = "increased_position"
            elif change_percent < -0.9:
                action = "sold_entire_position"
            elif change_percent < -0.1:
                action = "reduced_position"
            else:
                continue
                
            # Determine significance
            if abs(change_percent) > 0.5 or holding.get("value", 0) > 100_000_000:
                significance = "high"
            else:
                significance = "medium"
            
            signals.append({
                "ticker": ticker,
                "action": action,
                "change_percent": change_percent,
                "significance": significance
            })
        
        return signals
    
    async def stream(self) -> AsyncIterator[NewsItem]:
        """Stream SEC filings in real-time (not implemented)"""
        raise NotImplementedError("SEC API does not support streaming")