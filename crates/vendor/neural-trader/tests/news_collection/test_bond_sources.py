"""Tests for bond market news sources - RED phase"""

import pytest
from datetime import datetime
from aioresponses import aioresponses
import sys
sys.path.insert(0, '/workspaces/ai-news-trader/src')


@pytest.mark.asyncio
async def test_treasury_source_init():
    """Test Treasury news source initialization"""
    from news_trading.news_collection.sources.treasury_enhanced import TreasuryEnhancedSource
    
    source = TreasuryEnhancedSource()
    assert source.source_name == "treasury_direct"
    assert source.base_url == "https://api.treasurydirect.gov"


@pytest.mark.asyncio
async def test_treasury_auction_results():
    """Test parsing Treasury auction results"""
    from news_trading.news_collection.sources.treasury_enhanced import TreasuryEnhancedSource
    
    source = TreasuryEnhancedSource()
    
    with aioresponses() as mocked:
        mock_response = {
            "auctions": [
                {
                    "id": "treas-001",
                    "security_type": "10-Year Note",
                    "auction_date": "2024-01-15",
                    "issue_date": "2024-01-31",
                    "maturity_date": "2034-01-31",
                    "high_yield": "4.25",
                    "bid_to_cover": "2.58",
                    "indirect_bidders_pct": "68.5",
                    "direct_bidders_pct": "19.2",
                    "primary_dealers_pct": "12.3"
                }
            ]
        }
        
        mocked.get(
            "https://api.treasurydirect.gov/auctions/recent?limit=10",
            payload=mock_response
        )
        
        items = await source.fetch_latest(limit=10)
        assert len(items) == 1
        
        item = items[0]
        assert "10-Year Note" in item.title
        assert item.metadata["bond_type"] == "10-Year Note"
        assert item.metadata["yield"] == 4.25
        assert item.metadata["bid_to_cover"] == 2.58
        assert item.metadata["demand_strength"] == "strong"  # bid-to-cover > 2.5


@pytest.mark.asyncio
async def test_fed_announcement_parsing():
    """Test Federal Reserve announcement parsing"""
    from news_trading.news_collection.sources.federal_reserve_enhanced import FederalReserveEnhancedSource
    
    source = FederalReserveEnhancedSource()
    
    with aioresponses() as mocked:
        mock_response = {
            "announcements": [
                {
                    "id": "fed-001",
                    "title": "FOMC Statement: Federal Funds Rate Unchanged",
                    "content": "The Federal Open Market Committee decided to maintain the target range for the federal funds rate at 5.25-5.50 percent...",
                    "release_date": "2024-01-15T14:00:00Z",
                    "type": "monetary_policy",
                    "key_metrics": {
                        "fed_funds_rate": "5.25-5.50",
                        "decision": "unchanged",
                        "vote": "unanimous"
                    }
                }
            ]
        }
        
        # Import yarl
        from yarl import URL
        
        # Mock with query parameters
        url_with_params = str(URL("https://api.federalreserve.gov/announcements/recent").with_query({"limit": "100"}))
        mocked.get(
            url_with_params,
            payload=mock_response
        )
        
        items = await source.fetch_latest()
        assert len(items) == 1
        
        item = items[0]
        assert "FOMC" in item.title
        assert item.metadata["announcement_type"] == "monetary_policy"
        assert item.metadata["rate_decision"] == "unchanged"
        assert item.metadata["market_impact"] == "neutral"


@pytest.mark.asyncio
async def test_yield_curve_changes():
    """Test detection of significant yield curve changes"""
    from news_trading.news_collection.sources.yield_monitor import YieldCurveMonitor
    
    monitor = YieldCurveMonitor()
    
    # Mock yield curve data
    current_yields = {
        "3M": 5.45,
        "2Y": 4.35,
        "5Y": 4.10,
        "10Y": 4.25,
        "30Y": 4.45
    }
    
    previous_yields = {
        "3M": 5.40,
        "2Y": 4.45,
        "5Y": 4.20,
        "10Y": 4.15,
        "30Y": 4.35
    }
    
    analysis = monitor.analyze_yield_changes(current_yields, previous_yields)
    
    assert analysis["curve_shape"] == "inverted"  # 3M > 10Y
    assert analysis["steepening"] == True  # 10Y-2Y spread increased
    assert "10Y" in analysis["significant_moves"]
    assert abs(analysis["significant_moves"]["10Y"] - 0.10) < 0.01  # 10bp move
    assert analysis["trading_signal"] == "recession_hedge"  # Inverted curve = recession hedge


@pytest.mark.asyncio
async def test_tips_auction_analysis():
    """Test TIPS (Treasury Inflation-Protected Securities) auction analysis"""
    from news_trading.news_collection.sources.treasury_enhanced import TreasuryEnhancedSource
    
    source = TreasuryEnhancedSource()
    
    tips_auction = {
        "id": "tips-001",
        "security_type": "10-Year TIPS",
        "auction_date": "2024-01-15",
        "real_yield": "1.85",
        "bid_to_cover": "2.42",
        "breakeven_inflation": "2.40"  # Implied inflation = nominal yield - real yield
    }
    
    analysis = source._analyze_tips_auction(tips_auction)
    
    assert analysis["inflation_expectations"] == 2.40
    assert analysis["real_yield"] == 1.85
    assert analysis["market_sentiment"] == "neutral"  # 2.40% is in neutral range
    assert analysis["trading_opportunity"] == "balanced"  # Neutral sentiment = balanced