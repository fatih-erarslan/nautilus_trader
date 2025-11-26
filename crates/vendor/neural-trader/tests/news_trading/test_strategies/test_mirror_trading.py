"""Tests for Mirror Trading Strategy following TDD."""

import pytest
from datetime import datetime, timedelta
from src.news_trading.strategies.mirror_trading import MirrorTradingStrategy
from src.news_trading.decision_engine.models import (
    TradingSignal, SignalType, RiskLevel, TradingStrategy, AssetType
)


class TestMirrorTradingStrategy:
    """Test suite for Mirror Trading Strategy."""
    
    def test_mirror_strategy_initialization(self):
        """Test mirror trading strategy initialization."""
        strategy = MirrorTradingStrategy(portfolio_size=100000)
        
        assert strategy.portfolio_size == 100000
        assert "Berkshire Hathaway" in strategy.trusted_institutions
        assert strategy.trusted_institutions["Berkshire Hathaway"] == 0.95
        assert strategy.max_position_pct == 0.03  # 3% max per position
        
    def test_institutional_confidence_scoring(self):
        """Test confidence scoring for different institutions."""
        strategy = MirrorTradingStrategy()
        
        # Test high-confidence institutions
        berkshire_score = strategy.get_institution_confidence("Berkshire Hathaway")
        assert berkshire_score == 0.95
        
        # Test medium-confidence institutions
        tiger_score = strategy.get_institution_confidence("Tiger Global")
        assert tiger_score == 0.75
        
        # Test unknown institution
        unknown_score = strategy.get_institution_confidence("Unknown Fund")
        assert unknown_score == 0.50  # Default confidence
        
    def test_13f_filing_parser(self):
        """Test parsing of 13F filings."""
        strategy = MirrorTradingStrategy()
        
        filing_13f = {
            "filer": "Berkshire Hathaway",
            "quarter": "2024Q1",
            "holdings": [
                {"ticker": "AAPL", "shares": 900000000, "value": 150000000000},
                {"ticker": "BAC", "shares": 1000000000, "value": 35000000000},
                {"ticker": "CVX", "shares": 120000000, "value": 18000000000}
            ],
            "new_positions": ["OXY", "C"],
            "increased_positions": ["CVX"],
            "reduced_positions": ["AAPL"],
            "sold_positions": ["TSM"]
        }
        
        signals = strategy.parse_13f_filing(filing_13f)
        
        assert len(signals) > 0
        assert any(s["ticker"] == "OXY" and s["action"] == "buy" for s in signals)
        assert any(s["ticker"] == "TSM" and s["action"] == "sell" for s in signals)
        assert any(s["ticker"] == "CVX" and s["action"] == "buy" for s in signals)
        
        # Check confidence levels
        new_position_signal = next(s for s in signals if s["ticker"] == "OXY")
        assert new_position_signal["confidence"] == 0.95
        assert new_position_signal["priority"] == "high"
        
    def test_insider_transaction_scoring(self):
        """Test scoring of insider transactions."""
        strategy = MirrorTradingStrategy()
        
        # CEO buying - high confidence
        ceo_buy = {
            "filer": "Tim Cook",
            "company": "AAPL",
            "role": "CEO",
            "transaction_type": "Purchase",
            "shares": 50000,
            "price": 175.00
        }
        
        ceo_score = strategy.score_insider_transaction(ceo_buy)
        assert ceo_score > 0.85
        
        # Director selling - lower confidence
        director_sell = {
            "filer": "John Doe",
            "company": "MSFT",
            "role": "Director",
            "transaction_type": "Sale",
            "shares": 10000,
            "price": 400.00
        }
        
        director_score = strategy.score_insider_transaction(director_sell)
        assert director_score < 0.5  # Selling is negative signal
        
    def test_mirror_position_sizing(self):
        """Test position sizing based on institutional commitment."""
        strategy = MirrorTradingStrategy(portfolio_size=100000)
        
        # Buffett makes a big bet
        institutional_trade = {
            "institution": "Berkshire Hathaway",
            "ticker": "OXY",
            "action": "buy",
            "position_size_pct": 0.15,  # 15% of their portfolio
            "dollar_value": 15000000000
        }
        
        our_position = strategy.calculate_mirror_position(institutional_trade)
        
        assert our_position["size_pct"] <= 0.03  # Max 3% for safety
        assert our_position["size_pct"] > 0.02  # Should be significant
        assert our_position["confidence"] == 0.95
        assert "Scaling down" in our_position["reasoning"]
        
    def test_position_sizing_for_unknown_institution(self):
        """Test conservative sizing for unknown institutions."""
        strategy = MirrorTradingStrategy(portfolio_size=100000)
        
        unknown_trade = {
            "institution": "Small Fund LLC",
            "ticker": "GOOGL",
            "action": "buy",
            "position_size_pct": 0.20,
            "dollar_value": 5000000
        }
        
        our_position = strategy.calculate_mirror_position(unknown_trade)
        
        assert our_position["size_pct"] <= 0.01  # Very conservative
        assert our_position["confidence"] == 0.50
        
    @pytest.mark.asyncio
    async def test_generate_mirror_signal(self):
        """Test complete mirror trading signal generation."""
        strategy = MirrorTradingStrategy()
        
        filing_data = {
            "institution": "Berkshire Hathaway",
            "ticker": "BAC",
            "action": "buy",
            "shares": 50000000,
            "avg_price": 32.50,
            "position_change_pct": 0.20,
            "total_value": 1625000000,
            "filing_date": datetime.now() - timedelta(hours=12)
        }
        
        signal = await strategy.generate_mirror_signal(filing_data)
        
        assert signal is not None
        assert signal.asset == "BAC"
        assert signal.strategy == TradingStrategy.MIRROR
        assert signal.signal_type == SignalType.BUY
        assert signal.mirror_source == "Berkshire Hathaway"
        assert signal.confidence >= 0.9
        assert signal.holding_period == "6-24 months"  # Long-term like Buffett
        assert signal.position_size <= 0.03
        
    @pytest.mark.asyncio
    async def test_entry_timing_immediate(self):
        """Test immediate entry timing for recent filings."""
        strategy = MirrorTradingStrategy()
        
        filing_data = {
            "filing_date": datetime.now() - timedelta(hours=6),
            "ticker": "MSFT",
            "current_price": 400.00,
            "filing_price": 395.00,  # Price when institution bought
            "volume_since_filing": 15000000
        }
        
        timing = await strategy.determine_entry_timing(filing_data)
        
        assert timing["entry_strategy"] == "immediate"
        assert timing["max_chase_price"] == pytest.approx(401.75, rel=0.01)  # 1.5% chase limit
        assert timing["urgency"] == "high"
        
    @pytest.mark.asyncio
    async def test_entry_timing_wait(self):
        """Test waiting strategy for older filings."""
        strategy = MirrorTradingStrategy()
        
        filing_data = {
            "filing_date": datetime.now() - timedelta(days=3),
            "ticker": "GOOGL",
            "current_price": 145.00,
            "filing_price": 140.00,  # Already up 3.5%
            "volume_since_filing": 50000000
        }
        
        timing = await strategy.determine_entry_timing(filing_data)
        
        assert timing["entry_strategy"] == "wait_for_pullback"
        assert timing["target_entry"] < 145.00
        assert timing["urgency"] == "low"
        
    def test_estimate_holding_period(self):
        """Test holding period estimation based on institution style."""
        strategy = MirrorTradingStrategy()
        
        assert strategy.estimate_holding_period("Berkshire Hathaway") == "6-24 months"
        assert strategy.estimate_holding_period("Tiger Global") == "3-12 months"
        assert strategy.estimate_holding_period("Renaissance Technologies") == "1-6 months"
        assert strategy.estimate_holding_period("Unknown Fund") == "3-6 months"