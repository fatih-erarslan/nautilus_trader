"""
Tests for bond market news sources - Phase 2B
Following TDD approach: RED -> GREEN -> REFACTOR
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from aioresponses import aioresponses
import aiohttp

from src.news.models import NewsItem


class TestTreasurySource:
    """Tests for Treasury Direct news source implementation"""
    
    def test_treasury_source_init(self):
        """Test Treasury Direct source initialization"""
        from src.news.sources.treasury import TreasurySource
        
        source = TreasurySource()
        assert source.source_name == "treasury_direct"
        assert source.base_url == "https://api.treasurydirect.gov"
    
    @pytest.mark.asyncio
    async def test_treasury_fetch_latest(self):
        """Test fetching latest Treasury announcements"""
        from src.news.sources.treasury import TreasurySource
        
        source = TreasurySource()
        
        # Mock Treasury auction announcement
        with aioresponses() as mocked:
            mocked.get(
                "https://api.treasurydirect.gov/GA_FI_Announcements",
                payload={
                    "announcements": [{
                        "cusip": "912828ZX5",
                        "securityType": "Bill",
                        "securityTerm": "13-Week",
                        "auctionDate": "2024-01-15",
                        "issueDate": "2024-01-18",
                        "maturityDate": "2024-04-18",
                        "highYield": "5.25",
                        "allocationPercentage": "35.50",
                        "competitiveBidToCoverRatio": "2.85"
                    }]
                }
            )
            
            items = await source.fetch_latest(limit=10)
            assert len(items) == 1
            assert "13-Week Treasury Bill Auction" in items[0].title
            assert items[0].metadata["security_type"] == "Bill"
            assert items[0].metadata["yield"] == 5.25
            assert items[0].metadata["bid_to_cover"] == 2.85
            assert items[0].metadata["bond_type"] == "13-Week Treasury Bill"
    
    @pytest.mark.asyncio
    async def test_treasury_yield_tracking(self):
        """Test tracking Treasury yield changes"""
        from src.news.sources.treasury import TreasurySource
        
        source = TreasurySource()
        
        # Mock yield data
        with aioresponses() as mocked:
            mocked.get(
                "https://api.treasurydirect.gov/NP_WS_XMLYieldCurve",
                payload={
                    "entry": {
                        "content": {
                            "properties": {
                                "NEW_DATE": "2024-01-15",
                                "BC_1MONTH": "5.45",
                                "BC_3MONTH": "5.40",
                                "BC_6MONTH": "5.35",
                                "BC_1YEAR": "5.00",
                                "BC_2YEAR": "4.50",
                                "BC_5YEAR": "4.25",
                                "BC_10YEAR": "4.15",
                                "BC_30YEAR": "4.30"
                            }
                        }
                    }
                }
            )
            
            yield_data = await source.fetch_yield_curve()
            assert yield_data["10Y"] == 4.15
            assert yield_data["2Y"] == 4.50
            
            # Test yield curve inversion detection
            inversion = source.detect_yield_curve_inversion(yield_data)
            assert inversion["is_inverted"] == True
            assert inversion["2Y_10Y_spread"] == -0.35  # Negative spread indicates inversion
    
    @pytest.mark.asyncio
    async def test_treasury_auction_results(self):
        """Test parsing Treasury auction results"""
        from src.news.sources.treasury import TreasurySource
        
        source = TreasurySource()
        
        with aioresponses() as mocked:
            mocked.get(
                "https://api.treasurydirect.gov/GA_FI_Auction_Results?securityType=Note&limit=5",
                payload={
                    "results": [{
                        "cusip": "912828ZY3",
                        "securityType": "Note",
                        "securityTerm": "10-Year",
                        "highYield": "4.25",
                        "allocationPercentage": "24.10",
                        "competitiveBidToCoverRatio": "2.50",
                        "indirectBidderPercentage": "68.4",
                        "directBidderPercentage": "18.2",
                        "primaryDealerPercentage": "13.4",
                        "auctionDate": "2024-01-10"
                    }]
                }
            )
            
            results = await source.fetch_auction_results("Note", limit=5)
            assert len(results) == 1
            
            # Check auction quality metrics
            quality = source.assess_auction_quality(results[0])
            assert quality["demand_strength"] == "moderate"  # bid-to-cover 2.5
            assert quality["foreign_demand"] == "strong"  # indirect 68.4%


class TestFederalReserveSource:
    """Tests for Federal Reserve economic data source"""
    
    def test_fed_source_init(self):
        """Test Federal Reserve source initialization"""
        from src.news.sources.federal_reserve import FederalReserveSource
        
        source = FederalReserveSource(api_key="test-key")
        assert source.source_name == "federal_reserve"
        assert source.api_key == "test-key"
        assert source.base_url == "https://api.stlouisfed.org/fred"
    
    @pytest.mark.asyncio
    async def test_fed_economic_data(self):
        """Test fetching Federal Reserve economic data"""
        from src.news.sources.federal_reserve import FederalReserveSource
        
        source = FederalReserveSource(api_key="test-key")
        
        # Mock FRED API response
        with aioresponses() as mocked:
            # Mock interest rate data
            mocked.get(
                "https://api.stlouisfed.org/fred/series/observations?series_id=DFF&api_key=test-key&limit=1&sort_order=desc&file_type=json",
                payload={
                    "observations": [{
                        "date": "2024-01-15",
                        "value": "5.33"
                    }]
                }
            )
            
            # Mock inflation data
            mocked.get(
                "https://api.stlouisfed.org/fred/series/observations?series_id=CPIAUCSL&api_key=test-key&limit=2&sort_order=desc&file_type=json",
                payload={
                    "observations": [
                        {"date": "2024-01-01", "value": "310.326"},
                        {"date": "2023-12-01", "value": "309.685"}
                    ]
                }
            )
            
            data = await source.fetch_economic_indicators()
            
            assert data["fed_funds_rate"] == 5.33
            assert "inflation_rate" in data
            assert data["inflation_rate"] > 0  # Positive inflation
    
    @pytest.mark.asyncio
    async def test_fed_meeting_minutes(self):
        """Test parsing FOMC meeting minutes"""
        from src.news.sources.federal_reserve import FederalReserveSource
        
        source = FederalReserveSource(api_key="test-key")
        
        with aioresponses() as mocked:
            mocked.get(
                "https://www.federalreserve.gov/monetarypolicy/fomcminutes20240131.htm",
                body="""
                <html>
                <body>
                <div class="col-xs-12 col-sm-8 col-md-8">
                <p>Participants noted that inflation had eased but remained elevated.
                Many participants remarked that the policy rate was likely at or near its peak.
                Several participants mentioned risks to financial stability from commercial real estate.</p>
                </div>
                </body>
                </html>
                """,
                content_type="text/html"
            )
            
            minutes = await source.fetch_fomc_minutes()
            
            assert minutes["date"] == "2024-01-31"
            assert "inflation" in minutes["key_topics"]
            assert "policy rate" in minutes["key_topics"]
            assert minutes["sentiment"]["hawkish_score"] < minutes["sentiment"]["dovish_score"]
    
    @pytest.mark.asyncio
    async def test_fed_policy_impact_detection(self):
        """Test detection of Fed policy impact on bond markets"""
        from src.news.sources.federal_reserve import FederalReserveSource
        
        source = FederalReserveSource(api_key="test-key")
        
        # Test various Fed headlines
        headlines = [
            ("Fed Signals Potential Rate Cuts in Coming Months", "dovish", "positive"),
            ("Federal Reserve Maintains Hawkish Stance on Inflation", "hawkish", "negative"),
            ("FOMC Minutes Reveal Division on Rate Path", "neutral", "neutral"),
            ("Fed Chair Powell Hints at Extended Pause", "dovish", "positive")
        ]
        
        for headline, expected_sentiment, expected_impact in headlines:
            analysis = source.analyze_policy_statement(headline)
            assert analysis["sentiment"] == expected_sentiment
            assert analysis["bond_market_impact"] == expected_impact


class TestBondMarketIntegration:
    """Tests for bond market news integration"""
    
    @pytest.mark.asyncio
    async def test_bond_yield_change_detection(self):
        """Test detection of significant bond yield changes"""
        from src.news.sources.bond_market import detect_yield_changes
        
        current_yields = {
            "2Y": 4.50,
            "5Y": 4.25,
            "10Y": 4.15,
            "30Y": 4.30
        }
        
        previous_yields = {
            "2Y": 4.45,
            "5Y": 4.20,
            "10Y": 4.00,  # 15 basis point move
            "30Y": 4.28
        }
        
        changes = detect_yield_changes(current_yields, previous_yields)
        
        assert changes["10Y"]["change_bps"] == 15
        assert changes["10Y"]["is_significant"] == True  # >10 bps is significant
        assert changes["2Y"]["change_bps"] == 5
        assert changes["2Y"]["is_significant"] == False
    
    @pytest.mark.asyncio
    async def test_bond_trading_signals(self):
        """Test generation of bond trading signals from news"""
        from src.news.sources.bond_market import generate_bond_signals
        
        news_items = [
            NewsItem(
                id="fed-1",
                title="Fed Signals Rate Cuts Coming",
                content="Federal Reserve officials indicated...",
                source="federal_reserve",
                timestamp=datetime.now(),
                url="https://example.com",
                entities=["FED", "RATES"],
                metadata={"sentiment": "dovish", "impact": "positive"}
            ),
            NewsItem(
                id="treasury-1",
                title="10-Year Treasury Auction Shows Weak Demand",
                content="Today's 10-year Treasury auction...",
                source="treasury_direct",
                timestamp=datetime.now(),
                url="https://example.com",
                entities=["10Y", "TREASURY"],
                metadata={"bid_to_cover": 2.1, "demand_strength": "weak"}
            )
        ]
        
        signals = generate_bond_signals(news_items)
        
        assert len(signals) == 2
        assert signals[0]["action"] == "buy"  # Dovish Fed = buy bonds
        assert signals[0]["instrument"] == "10Y_Treasury"
        assert signals[1]["action"] == "wait"  # Weak auction = wait for better entry