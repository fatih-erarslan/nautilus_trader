#!/usr/bin/env python3
"""
Senator Trade Data Scraper

Simulates fetching senate financial disclosures with realistic trading patterns.
Includes top performers like Pelosi, Tuberville, and other senators with 
historically notable trading activity.
"""

import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TransactionType(Enum):
    PURCHASE = "purchase"
    SALE = "sale"
    EXCHANGE = "exchange"

class AssetType(Enum):
    STOCK = "stock"
    BOND = "bond"
    MUTUAL_FUND = "mutual_fund"
    ETF = "etf"
    OPTION = "option"
    CRYPTO = "crypto"

@dataclass
class SenatorTrade:
    """Represents a single senator trading transaction"""
    senator_name: str
    party: str
    state: str
    transaction_date: str
    disclosure_date: str
    transaction_type: str
    asset_type: str
    asset_name: str
    ticker: Optional[str]
    amount_range: str  # e.g., "$1,001 - $15,000"
    amount_min: float
    amount_max: float
    amount_estimated: float
    committee_assignments: List[str]
    conflict_score: float  # 0-10 scale for potential conflicts of interest
    performance_score: float  # Estimated performance vs market
    filing_status: str  # "on_time", "late", "amended"

@dataclass
class SenatorProfile:
    """Profile information for a senator including trading patterns"""
    name: str
    party: str
    state: str
    committees: List[str]
    trading_frequency: str  # "high", "medium", "low"
    avg_performance: float  # Annual performance estimate
    specialty_sectors: List[str]
    notable_positions: List[str]

class SenatorTradeDataScraper:
    """
    Simulates scraping senator financial disclosure data with realistic patterns
    """
    
    def __init__(self):
        self.senator_profiles = self._initialize_senator_profiles()
        self.popular_stocks = self._get_popular_stocks()
        self.sector_etfs = self._get_sector_etfs()
        
    def _initialize_senator_profiles(self) -> List[SenatorProfile]:
        """Initialize senator profiles with realistic data"""
        return [
            # Top performers based on historical data
            SenatorProfile(
                name="Nancy Pelosi",
                party="Democrat",
                state="California", 
                committees=["House Financial Services", "House Intelligence"],
                trading_frequency="high",
                avg_performance=0.245,  # 24.5% annual return
                specialty_sectors=["Technology", "Healthcare", "Finance"],
                notable_positions=["NVDA", "TSLA", "AMZN", "GOOGL", "AAPL"]
            ),
            SenatorProfile(
                name="Tommy Tuberville",
                party="Republican",
                state="Alabama",
                committees=["Senate Armed Services", "Senate Agriculture"],
                trading_frequency="high", 
                avg_performance=0.189,  # 18.9% annual return
                specialty_sectors=["Defense", "Agriculture", "Energy"],
                notable_positions=["LMT", "RTX", "BA", "XLE", "DBA"]
            ),
            SenatorProfile(
                name="Dan Crenshaw",
                party="Republican",
                state="Texas",
                committees=["House Energy and Commerce", "House Budget"],
                trading_frequency="medium",
                avg_performance=0.156,
                specialty_sectors=["Energy", "Defense", "Technology"],
                notable_positions=["XOM", "CVX", "LMT", "MSFT"]
            ),
            SenatorProfile(
                name="Austin Scott",
                party="Republican", 
                state="Georgia",
                committees=["House Armed Services", "House Agriculture"],
                trading_frequency="medium",
                avg_performance=0.142,
                specialty_sectors=["Agriculture", "Defense"],
                notable_positions=["ADM", "CAT", "DE", "LMT"]
            ),
            SenatorProfile(
                name="Brian Mast",
                party="Republican",
                state="Florida", 
                committees=["House Foreign Affairs", "House Transportation"],
                trading_frequency="medium",
                avg_performance=0.138,
                specialty_sectors=["Defense", "Infrastructure"],
                notable_positions=["RTX", "CAT", "UNP", "CSX"]
            ),
            SenatorProfile(
                name="John Hickenlooper",
                party="Democrat",
                state="Colorado",
                committees=["Senate Commerce", "Senate Energy"],
                trading_frequency="low",
                avg_performance=0.089,
                specialty_sectors=["Energy", "Technology"],
                notable_positions=["TSLA", "ENPH", "NEE"]
            ),
            SenatorProfile(
                name="Sheldon Whitehouse",
                party="Democrat",
                state="Rhode Island",
                committees=["Senate Judiciary", "Senate Environment"],
                trading_frequency="low",
                avg_performance=0.076,
                specialty_sectors=["Clean Energy", "Healthcare"],
                notable_positions=["ICLN", "PFE", "JNJ"]
            ),
            # Additional senators with varied performance
            SenatorProfile(
                name="Josh Hawley",
                party="Republican", 
                state="Missouri",
                committees=["Senate Judiciary", "Senate Armed Services"],
                trading_frequency="low",
                avg_performance=0.034,
                specialty_sectors=["Technology", "Finance"],
                notable_positions=["BRK-B", "JPM", "BAC"]
            ),
            SenatorProfile(
                name="Elizabeth Warren",
                party="Democrat",
                state="Massachusetts", 
                committees=["Senate Banking", "Senate Armed Services"],
                trading_frequency="low",
                avg_performance=-0.012,  # Negative performance
                specialty_sectors=["Finance", "Healthcare"],
                notable_positions=["VTI", "VXUS"]  # Index funds
            ),
            SenatorProfile(
                name="Ted Cruz",
                party="Republican",
                state="Texas",
                committees=["Senate Judiciary", "Senate Commerce"],
                trading_frequency="medium", 
                avg_performance=0.067,
                specialty_sectors=["Energy", "Technology"],
                notable_positions=["XOM", "TSLA", "BTC-USD"]
            )
        ]
    
    def _get_popular_stocks(self) -> List[Dict[str, Any]]:
        """Get list of popular stocks traded by senators"""
        return [
            {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
            {"ticker": "MSFT", "name": "Microsoft Corp.", "sector": "Technology"},
            {"ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology"},
            {"ticker": "AMZN", "name": "Amazon.com Inc.", "sector": "Technology"},
            {"ticker": "TSLA", "name": "Tesla Inc.", "sector": "Automotive/Energy"},
            {"ticker": "NVDA", "name": "NVIDIA Corp.", "sector": "Technology"},
            {"ticker": "META", "name": "Meta Platforms Inc.", "sector": "Technology"},
            {"ticker": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Finance"},
            {"ticker": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare"},
            {"ticker": "PFE", "name": "Pfizer Inc.", "sector": "Healthcare"},
            {"ticker": "XOM", "name": "Exxon Mobil Corp.", "sector": "Energy"},
            {"ticker": "CVX", "name": "Chevron Corp.", "sector": "Energy"},
            {"ticker": "LMT", "name": "Lockheed Martin Corp.", "sector": "Defense"},
            {"ticker": "RTX", "name": "Raytheon Technologies", "sector": "Defense"},
            {"ticker": "BA", "name": "Boeing Co.", "sector": "Defense/Aerospace"},
            {"ticker": "CAT", "name": "Caterpillar Inc.", "sector": "Industrial"},
            {"ticker": "DE", "name": "Deere & Co.", "sector": "Agriculture/Industrial"},
            {"ticker": "ADM", "name": "Archer-Daniels-Midland", "sector": "Agriculture"},
            {"ticker": "BRK-B", "name": "Berkshire Hathaway Inc.", "sector": "Conglomerate"},
            {"ticker": "SPY", "name": "SPDR S&P 500 ETF", "sector": "ETF"}
        ]
    
    def _get_sector_etfs(self) -> List[Dict[str, Any]]:
        """Get sector-specific ETFs commonly traded"""
        return [
            {"ticker": "XLE", "name": "Energy Select Sector SPDR", "sector": "Energy"},
            {"ticker": "XLF", "name": "Financial Select Sector SPDR", "sector": "Finance"}, 
            {"ticker": "XLK", "name": "Technology Select Sector SPDR", "sector": "Technology"},
            {"ticker": "XLV", "name": "Health Care Select Sector SPDR", "sector": "Healthcare"},
            {"ticker": "XLI", "name": "Industrial Select Sector SPDR", "sector": "Industrial"},
            {"ticker": "ICLN", "name": "iShares Global Clean Energy ETF", "sector": "Clean Energy"},
            {"ticker": "VTI", "name": "Vanguard Total Stock Market ETF", "sector": "Broad Market"},
            {"ticker": "VXUS", "name": "Vanguard Total International Stock ETF", "sector": "International"}
        ]
    
    def _generate_amount_range(self, estimated_amount: float) -> tuple:
        """Generate realistic disclosure amount ranges"""
        ranges = [
            (1000, 15000, "$1,001 - $15,000"),
            (15001, 50000, "$15,001 - $50,000"), 
            (50001, 100000, "$50,001 - $100,000"),
            (100001, 250000, "$100,001 - $250,000"),
            (250001, 500000, "$250,001 - $500,000"),
            (500001, 1000000, "$500,001 - $1,000,000"),
            (1000001, 5000000, "$1,000,001 - $5,000,000"),
            (5000001, float('inf'), "$5,000,001+")
        ]
        
        for min_amt, max_amt, range_str in ranges:
            if min_amt <= estimated_amount <= max_amt:
                return min_amt, max_amt, range_str
                
        # Default fallback
        return 1000, 15000, "$1,001 - $15,000"
    
    def _calculate_conflict_score(self, senator: SenatorProfile, asset_sector: str) -> float:
        """Calculate potential conflict of interest score"""
        base_score = random.uniform(0.5, 2.0)
        
        # Higher scores for committee relevance
        committee_conflicts = {
            "House Financial Services": ["Finance", "Banking"],
            "Senate Banking": ["Finance", "Banking"],
            "House Energy and Commerce": ["Energy", "Healthcare", "Technology"],
            "Senate Energy": ["Energy"],
            "House Armed Services": ["Defense", "Aerospace"],
            "Senate Armed Services": ["Defense", "Aerospace"],
            "House Agriculture": ["Agriculture"],
            "Senate Agriculture": ["Agriculture"],
            "House Intelligence": ["Technology", "Defense"],
            "Senate Commerce": ["Technology", "Transportation"]
        }
        
        for committee in senator.committees:
            if committee in committee_conflicts:
                if asset_sector in committee_conflicts[committee]:
                    base_score += random.uniform(2.0, 4.0)
                    
        return min(base_score, 10.0)
    
    def _generate_trade_for_senator(self, senator: SenatorProfile, trade_date: datetime) -> SenatorTrade:
        """Generate a realistic trade for a specific senator"""
        
        # Choose asset based on senator's specialty
        all_assets = self.popular_stocks + self.sector_etfs
        
        # Weight selection toward senator's specialty sectors
        if senator.specialty_sectors and random.random() < 0.6:
            specialty_assets = [asset for asset in all_assets 
                             if asset['sector'] in senator.specialty_sectors]
            if specialty_assets:
                chosen_asset = random.choice(specialty_assets)
            else:
                chosen_asset = random.choice(all_assets)
        else:
            chosen_asset = random.choice(all_assets)
        
        # Transaction type probabilities
        transaction_type = random.choices(
            [TransactionType.PURCHASE, TransactionType.SALE],
            weights=[0.6, 0.4]
        )[0]
        
        # Asset type
        if chosen_asset['ticker'] in ['SPY', 'VTI', 'VXUS', 'XLE', 'XLF', 'XLK', 'XLV', 'XLI', 'ICLN']:
            asset_type = AssetType.ETF
        else:
            asset_type = AssetType.STOCK
            
        # Amount based on senator's typical trading patterns
        if senator.trading_frequency == "high":
            base_amount = random.uniform(50000, 500000)
        elif senator.trading_frequency == "medium": 
            base_amount = random.uniform(15000, 100000)
        else:
            base_amount = random.uniform(1000, 50000)
            
        # Adjust for notable positions
        if chosen_asset['ticker'] in senator.notable_positions:
            base_amount *= random.uniform(1.5, 3.0)
            
        amount_min, amount_max, amount_range = self._generate_amount_range(base_amount)
        
        # Performance score based on senator's historical performance
        performance_multiplier = 1 + (senator.avg_performance * random.uniform(0.8, 1.2))
        performance_score = performance_multiplier * random.uniform(0.9, 1.1)
        
        # Filing status - most on time, some late
        filing_status = random.choices(
            ["on_time", "late", "amended"],
            weights=[0.75, 0.20, 0.05]
        )[0]
        
        # Disclosure date (30-45 days after transaction for "on_time")
        if filing_status == "on_time":
            disclosure_delay = random.randint(30, 45)
        elif filing_status == "late":
            disclosure_delay = random.randint(46, 120)  
        else:  # amended
            disclosure_delay = random.randint(35, 200)
            
        disclosure_date = trade_date + timedelta(days=disclosure_delay)
        
        return SenatorTrade(
            senator_name=senator.name,
            party=senator.party,
            state=senator.state,
            transaction_date=trade_date.strftime("%Y-%m-%d"),
            disclosure_date=disclosure_date.strftime("%Y-%m-%d"),
            transaction_type=transaction_type.value,
            asset_type=asset_type.value,
            asset_name=chosen_asset['name'],
            ticker=chosen_asset['ticker'],
            amount_range=amount_range,
            amount_min=amount_min,
            amount_max=amount_max,
            amount_estimated=base_amount,
            committee_assignments=senator.committees,
            conflict_score=self._calculate_conflict_score(senator, chosen_asset['sector']),
            performance_score=performance_score,
            filing_status=filing_status
        )
    
    def generate_mock_data(self, days_back: int = 365, trades_per_senator: int = None) -> List[SenatorTrade]:
        """
        Generate mock senator trading data
        
        Args:
            days_back: Number of days back to generate data for
            trades_per_senator: Override for number of trades per senator
        """
        logger.info(f"Generating mock senator trading data for {days_back} days")
        
        trades = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        for senator in self.senator_profiles:
            # Determine number of trades based on trading frequency
            if trades_per_senator:
                num_trades = trades_per_senator
            else:
                if senator.trading_frequency == "high":
                    num_trades = random.randint(15, 35)
                elif senator.trading_frequency == "medium":
                    num_trades = random.randint(5, 15) 
                else:
                    num_trades = random.randint(1, 8)
            
            # Generate trades spread over the time period
            for _ in range(num_trades):
                # Random date within the period
                days_offset = random.randint(0, days_back)
                trade_date = start_date + timedelta(days=days_offset)
                
                trade = self._generate_trade_for_senator(senator, trade_date)
                trades.append(trade)
        
        # Sort by disclosure date (most recent first)
        trades.sort(key=lambda x: x.disclosure_date, reverse=True)
        
        logger.info(f"Generated {len(trades)} trades across {len(self.senator_profiles)} senators")
        return trades
    
    def save_to_json(self, trades: List[SenatorTrade], filename: str = "senator_trades.json") -> str:
        """Save trades data to JSON file"""
        filepath = f"/workspaces/neural-trader/src/{filename}"
        
        # Convert to dictionaries for JSON serialization
        trades_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_trades": len(trades),
                "senators_count": len(set(trade.senator_name for trade in trades)),
                "date_range": {
                    "earliest_trade": min(trade.transaction_date for trade in trades),
                    "latest_trade": max(trade.transaction_date for trade in trades),
                    "earliest_disclosure": min(trade.disclosure_date for trade in trades),
                    "latest_disclosure": max(trade.disclosure_date for trade in trades)
                }
            },
            "trades": [asdict(trade) for trade in trades]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(trades_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved {len(trades)} trades to {filepath}")
        return filepath
    
    def get_performance_summary(self, trades: List[SenatorTrade]) -> Dict[str, Any]:
        """Generate performance summary statistics"""
        
        senator_stats = {}
        for trade in trades:
            if trade.senator_name not in senator_stats:
                senator_stats[trade.senator_name] = {
                    'trades': 0,
                    'total_estimated_value': 0,
                    'avg_performance': 0,
                    'avg_conflict_score': 0,
                    'party': trade.party,
                    'state': trade.state
                }
            
            stats = senator_stats[trade.senator_name]
            stats['trades'] += 1
            stats['total_estimated_value'] += trade.amount_estimated
            stats['avg_performance'] = ((stats['avg_performance'] * (stats['trades'] - 1)) + 
                                      trade.performance_score) / stats['trades']
            stats['avg_conflict_score'] = ((stats['avg_conflict_score'] * (stats['trades'] - 1)) + 
                                          trade.conflict_score) / stats['trades']
        
        # Sort by estimated performance
        top_performers = sorted(senator_stats.items(), 
                               key=lambda x: x[1]['avg_performance'], 
                               reverse=True)
        
        return {
            "summary": {
                "total_senators": len(senator_stats),
                "total_trades": len(trades),
                "total_estimated_volume": sum(s['total_estimated_value'] for s in senator_stats.values()),
                "avg_trade_size": sum(s['total_estimated_value'] for s in senator_stats.values()) / len(trades)
            },
            "top_performers": {name: stats for name, stats in top_performers[:10]},
            "by_party": {
                "Republican": {
                    "senators": len([s for s in senator_stats.values() if s['party'] == 'Republican']),
                    "avg_performance": sum(s['avg_performance'] for s in senator_stats.values() 
                                         if s['party'] == 'Republican') / 
                                     len([s for s in senator_stats.values() if s['party'] == 'Republican'])
                },
                "Democrat": {
                    "senators": len([s for s in senator_stats.values() if s['party'] == 'Democrat']),
                    "avg_performance": sum(s['avg_performance'] for s in senator_stats.values() 
                                         if s['party'] == 'Democrat') / 
                                     len([s for s in senator_stats.values() if s['party'] == 'Democrat'])
                }
            }
        }

def main():
    """Main function to demonstrate the scraper"""
    
    print("🏛️  Senator Trade Data Scraper")
    print("=" * 50)
    
    scraper = SenatorTradeDataScraper()
    
    # Generate mock data
    trades = scraper.generate_mock_data(days_back=365)
    
    # Save to JSON
    json_file = scraper.save_to_json(trades)
    
    # Generate performance summary
    summary = scraper.get_performance_summary(trades)
    
    # Display summary
    print(f"\n📊 Summary Statistics:")
    print(f"Total Senators: {summary['summary']['total_senators']}")
    print(f"Total Trades: {summary['summary']['total_trades']}")
    print(f"Total Estimated Volume: ${summary['summary']['total_estimated_volume']:,.0f}")
    print(f"Average Trade Size: ${summary['summary']['avg_trade_size']:,.0f}")
    
    print(f"\n🏆 Top Performers by Estimated Returns:")
    for i, (name, stats) in enumerate(list(summary['top_performers'].items())[:5], 1):
        print(f"{i}. {name} ({stats['party']}) - {stats['avg_performance']:.1%} avg return")
        print(f"   Trades: {stats['trades']}, Volume: ${stats['total_estimated_value']:,.0f}")
    
    print(f"\n🎉 Data saved to: {json_file}")
    
    return json_file

if __name__ == "__main__":
    main()