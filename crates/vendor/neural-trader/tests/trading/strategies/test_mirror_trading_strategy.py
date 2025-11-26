"""Test suite for Mirror Trading Strategy implementation."""

import pytest
from datetime import datetime, timedelta
from src.trading.strategies.mirror_trader import MirrorTradingEngine


class TestMirrorTradingStrategy:
    """Test cases for mirror trading strategy."""
    
    def test_institutional_filing_parser(self):
        """Test parsing of 13F and Form 4 filings."""
        engine = MirrorTradingEngine()
        
        # Mock 13F filing data
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
        
        signals = engine.parse_13f_filing(filing_13f)
        
        assert len(signals) > 0
        assert any(s["ticker"] == "OXY" and s["action"] == "buy" for s in signals)
        assert any(s["ticker"] == "TSM" and s["action"] == "sell" for s in signals)
        assert any(s["ticker"] == "CVX" and s["action"] == "buy" for s in signals)
        
        # Check signal details
        oxy_signal = next(s for s in signals if s["ticker"] == "OXY")
        assert oxy_signal["priority"] == "high"
        assert oxy_signal["mirror_type"] == "new_position"
        assert abs(oxy_signal["confidence"] - 0.7052) < 0.01  # Optimized Berkshire confidence
        
        tsm_signal = next(s for s in signals if s["ticker"] == "TSM")
        assert tsm_signal["priority"] == "high"
        assert tsm_signal["mirror_type"] == "exit_position"
        
    def test_confidence_scoring_by_institution(self):
        """Test confidence scoring based on institution track record."""
        engine = MirrorTradingEngine()
        
        # Test optimized confidence institutions
        berkshire_score = engine.get_institution_confidence("Berkshire Hathaway")
        assert abs(berkshire_score - 0.7052) < 0.01  # Optimized value
        
        bridgewater_score = engine.get_institution_confidence("Bridgewater Associates")
        assert abs(bridgewater_score - 0.9047) < 0.01  # Optimized value
        
        renaissance_score = engine.get_institution_confidence("Renaissance Technologies")
        assert abs(renaissance_score - 0.9307) < 0.01  # Optimized value
        
        # Test unknown institution
        unknown_score = engine.get_institution_confidence("Unknown Fund")
        assert unknown_score == 0.5  # Default score
        
        # Test insider transactions
        insider_filing = {
            "filer": "Tim Cook",
            "company": "AAPL",
            "role": "CEO",
            "transaction_type": "Purchase",
            "shares": 50000,
            "price": 175.00,
            "transaction_value": 8750000
        }
        
        insider_score = engine.score_insider_transaction(insider_filing)
        assert insider_score > 0.85  # CEO buying is high confidence
        
        # Test insider selling
        insider_sale = {
            "filer": "CFO Name",
            "company": "XYZ",
            "role": "CFO",
            "transaction_type": "Sale",
            "shares": 10000,
            "price": 50.00,
            "transaction_value": 500000
        }
        
        sale_score = engine.score_insider_transaction(insider_sale)
        assert sale_score < 0.4  # Insider selling is negative signal
        
    def test_mirror_position_sizing(self):
        """Test position sizing based on institutional commitment."""
        engine = MirrorTradingEngine(portfolio_size=100000)
        
        # Buffett makes a big bet
        institutional_trade = {
            "institution": "Berkshire Hathaway",
            "ticker": "OXY",
            "action": "buy",
            "position_size_pct": 0.15,  # 15% of their portfolio
            "dollar_value": 15000000000
        }
        
        our_position = engine.calculate_mirror_position(institutional_trade)
        
        # We should take a proportional but smaller position
        assert our_position["size_pct"] <= 0.0329  # Optimized max position
        assert our_position["reasoning"] == "Scaling down institutional position for risk management"
        assert abs(our_position["confidence"] - 0.7052) < 0.01  # Optimized Berkshire confidence
        assert our_position["expected_holding_period"] == "6-24 months"
        
        # Test smaller institution
        smaller_trade = {
            "institution": "Tiger Global",
            "ticker": "MSFT",
            "action": "buy",
            "position_size_pct": 0.08,
            "dollar_value": 2000000000
        }
        
        smaller_position = engine.calculate_mirror_position(smaller_trade)
        assert smaller_position["size_pct"] < our_position["size_pct"]  # Lower confidence = smaller position
        assert abs(smaller_position["confidence"] - 0.6529) < 0.01  # Optimized Tiger Global confidence
        
    @pytest.mark.asyncio
    async def test_timing_mirror_trades(self):
        """Test entry timing for mirror trades."""
        engine = MirrorTradingEngine()
        
        # Recent filing - good timing
        filing_data = {
            "filing_date": datetime.now() - timedelta(hours=6),
            "ticker": "MSFT",
            "current_price": 400.00,
            "filing_price": 395.00,  # Price when institution bought
            "volume_since_filing": 15000000,
            "days_since_filing": 0.25
        }
        
        timing = await engine.determine_entry_timing(filing_data)
        
        assert timing["entry_strategy"] == "immediate"  # Still close to filing price
        assert abs(timing["max_chase_price"] - 400.925) < 0.01  # Don't chase more than 1.5% (395 * 1.015)
        assert timing["urgency"] == "high"
        
        # Old filing - less urgency
        old_filing_data = {
            "filing_date": datetime.now() - timedelta(days=10),
            "ticker": "GOOGL",
            "current_price": 150.00,
            "filing_price": 140.00,
            "volume_since_filing": 50000000,
            "days_since_filing": 10
        }
        
        old_timing = await engine.determine_entry_timing(old_filing_data)
        
        assert old_timing["entry_strategy"] == "patient"
        assert old_timing["urgency"] == "low"
        
        # Price moved too much
        expensive_filing_data = {
            "filing_date": datetime.now() - timedelta(hours=12),
            "ticker": "TSLA",
            "current_price": 250.00,
            "filing_price": 200.00,  # 25% higher now
            "volume_since_filing": 30000000,
            "days_since_filing": 0.5
        }
        
        expensive_timing = await engine.determine_entry_timing(expensive_filing_data)
        
        assert expensive_timing["entry_strategy"] == "wait_for_pullback"
        assert expensive_timing["urgency"] == "low"
        
    def test_form_4_insider_parsing(self):
        """Test parsing of Form 4 insider transactions."""
        engine = MirrorTradingEngine()
        
        # CEO purchase
        form_4_data = {
            "filer": "Elon Musk",
            "company": "TSLA",
            "ticker": "TSLA",
            "role": "CEO",
            "transaction_type": "Purchase",
            "shares": 100000,
            "price": 200.00,
            "transaction_date": datetime.now() - timedelta(days=1),
            "ownership_pct": 0.13
        }
        
        insider_signal = engine.parse_form_4_filing(form_4_data)
        
        assert insider_signal["action"] == "buy"
        assert insider_signal["confidence"] > 0.8
        assert insider_signal["signal_strength"] == "strong"
        assert insider_signal["position_size_multiplier"] > 1.0  # CEO buying gets boost
        
        # Director sale (less negative than executive sale)
        director_sale = {
            "filer": "Board Member",
            "company": "AAPL",
            "ticker": "AAPL",
            "role": "Director",
            "transaction_type": "Sale",
            "shares": 5000,
            "price": 175.00,
            "transaction_date": datetime.now() - timedelta(days=2),
            "ownership_pct": 0.001
        }
        
        director_signal = engine.parse_form_4_filing(director_sale)
        
        assert director_signal["action"] == "neutral"  # Less concerning than exec sale
        assert director_signal["confidence"] < 0.6
        
    def test_institutional_track_record_analysis(self):
        """Test analysis of institutional track record."""
        engine = MirrorTradingEngine()
        
        # Mock track record data
        track_record = {
            "institution": "Berkshire Hathaway",
            "last_5_years": {
                "annual_returns": [0.12, 0.08, 0.15, 0.06, 0.18],
                "winning_positions": 145,
                "total_positions": 180,
                "avg_holding_period": 18  # months
            },
            "recent_performance": {
                "last_12_months": 0.11,
                "vs_sp500": 0.03  # 3% outperformance
            }
        }
        
        analysis = engine.analyze_institutional_track_record(track_record)
        
        assert analysis["confidence_score"] > 0.86
        assert analysis["win_rate"] > 0.8
        assert analysis["consistency_score"] > 0.59
        assert analysis["recommended_follow_pct"] > 0.8  # High percentage to follow
        
        # Test poor track record
        poor_record = {
            "institution": "Struggling Fund",
            "last_5_years": {
                "annual_returns": [-0.05, 0.02, -0.08, 0.01, -0.12],
                "winning_positions": 45,
                "total_positions": 120,
                "avg_holding_period": 3  # months - short term focus
            },
            "recent_performance": {
                "last_12_months": -0.08,
                "vs_sp500": -0.15  # 15% underperformance
            }
        }
        
        poor_analysis = engine.analyze_institutional_track_record(poor_record)
        
        assert poor_analysis["confidence_score"] < 0.4
        assert poor_analysis["recommended_follow_pct"] <= 0.3
        
    def test_portfolio_overlap_analysis(self):
        """Test analysis of portfolio overlap with institutions."""
        engine = MirrorTradingEngine()
        
        our_portfolio = {
            "AAPL": 0.15,  # 15% of our portfolio
            "MSFT": 0.12,
            "GOOGL": 0.08,
            "TSLA": 0.05,
            "NVDA": 0.10
        }
        
        institutional_portfolio = {
            "AAPL": 0.25,  # 25% of their portfolio
            "MSFT": 0.18,
            "AMZN": 0.15,  # They have, we don't
            "TSLA": 0.03,  # They have less
            "META": 0.12   # They have, we don't
        }
        
        overlap_analysis = engine.analyze_portfolio_overlap(our_portfolio, institutional_portfolio)
        
        assert overlap_analysis["overlap_pct"] > 0.3  # Significant overlap
        assert "AMZN" in overlap_analysis["missing_positions"]
        assert "META" in overlap_analysis["missing_positions"]
        assert "NVDA" in overlap_analysis["our_unique_positions"]
        
        # Test recommendations
        recommendations = overlap_analysis["recommendations"]
        assert any(r["ticker"] == "AMZN" and r["action"] == "consider_buy" for r in recommendations)
        
    def test_mirror_risk_management(self):
        """Test risk management for mirror trading positions."""
        engine = MirrorTradingEngine()
        
        # Test position with institutional exit
        position_with_exit = {
            "ticker": "XYZ",
            "entry_price": 100.00,
            "current_price": 110.00,
            "institution": "Berkshire Hathaway",
            "institutional_status": "exited",  # Institution sold
            "days_held": 45,
            "our_position_pct": 0.02
        }
        
        risk_assessment = engine.assess_mirror_risk(position_with_exit)
        
        assert risk_assessment["action"] == "exit"
        assert risk_assessment["reason"] == "institutional_exit"
        assert risk_assessment["urgency"] == "high"
        
        # Test position with institutional increase
        position_with_increase = {
            "ticker": "ABC",
            "entry_price": 50.00,
            "current_price": 45.00,  # Down 10%
            "institution": "Tiger Global",
            "institutional_status": "increased",  # Institution added more
            "days_held": 30,
            "our_position_pct": 0.015
        }
        
        increased_assessment = engine.assess_mirror_risk(position_with_increase)
        
        assert increased_assessment["action"] == "hold"  # Institution doubling down
        assert increased_assessment["reason"] == "institutional_confidence"
        
    def test_sector_institutional_flow(self):
        """Test tracking institutional flows by sector."""
        engine = MirrorTradingEngine()
        
        sector_flows = {
            "Technology": {
                "net_flow": 5000000000,  # $5B net inflow
                "major_buyers": ["Berkshire Hathaway", "Tiger Global"],
                "major_sellers": ["Soros Fund Management"],
                "flow_consistency": 0.8  # 80% of institutions buying
            },
            "Energy": {
                "net_flow": -2000000000,  # $2B net outflow
                "major_buyers": ["Pershing Square"],
                "major_sellers": ["Bridgewater Associates", "Renaissance Technologies"],
                "flow_consistency": 0.7
            },
            "Healthcare": {
                "net_flow": 1000000000,  # $1B net inflow
                "major_buyers": ["Third Point"],
                "major_sellers": [],
                "flow_consistency": 0.6
            }
        }
        
        flow_analysis = engine.analyze_sector_flows(sector_flows)
        
        assert flow_analysis["strongest_inflow_sector"] == "Technology"
        assert flow_analysis["strongest_outflow_sector"] == "Energy"
        assert "Technology" in flow_analysis["recommended_sectors"]
        assert "Energy" in flow_analysis["avoid_sectors"]
        
    def test_mirror_performance_tracking(self):
        """Test performance tracking for mirror trades."""
        engine = MirrorTradingEngine()
        
        mirror_trades = [
            {
                "ticker": "AAPL",
                "institution": "Berkshire Hathaway",
                "entry_date": datetime.now() - timedelta(days=90),
                "entry_price": 150.00,
                "current_price": 175.00,
                "institutional_entry": 148.00,
                "institutional_current": 175.00
            },
            {
                "ticker": "TSLA",
                "institution": "Tiger Global",
                "entry_date": datetime.now() - timedelta(days=60),
                "entry_price": 200.00,
                "current_price": 180.00,
                "institutional_entry": 195.00,
                "institutional_current": 180.00
            }
        ]
        
        performance = engine.track_mirror_performance(mirror_trades)
        
        assert performance["our_avg_return"] > 0  # Should be positive overall
        assert performance["institutional_avg_return"] > 0
        assert performance["tracking_efficiency"] > 0.8  # We should track institutions well
        
        # Check individual trade tracking
        aapl_performance = next(p for p in performance["individual_trades"] if p["ticker"] == "AAPL")
        assert aapl_performance["our_return"] > 0.15  # ~16.7% return
        assert aapl_performance["institutional_return"] > 0.15
        
    @pytest.mark.asyncio
    async def test_real_time_filing_monitor(self):
        """Test real-time monitoring of institutional filings."""
        engine = MirrorTradingEngine()
        
        # Mock recent filings
        recent_filings = [
            {
                "institution": "Berkshire Hathaway",
                "filing_type": "13F",
                "filed_date": datetime.now() - timedelta(hours=2),
                "new_positions": ["BRK.B"],  # Self-purchase
                "position_changes": {"OXY": "increased", "AAPL": "reduced"}
            },
            {
                "institution": "Renaissance Technologies",
                "filing_type": "13F-HR",
                "filed_date": datetime.now() - timedelta(hours=6),
                "new_positions": ["NVDA", "AMD"],
                "position_changes": {"GOOGL": "exited"}
            }
        ]
        
        alerts = await engine.monitor_filing_alerts(recent_filings)
        
        assert len(alerts) > 0
        assert any(a["ticker"] == "OXY" and a["action"] == "buy" for a in alerts)
        assert any(a["ticker"] == "GOOGL" and a["action"] == "sell" for a in alerts)
        
        # Check alert priorities
        high_priority_alerts = [a for a in alerts if a["priority"] == "high"]
        assert len(high_priority_alerts) > 0
        
    def test_mirror_exit_strategies(self):
        """Test exit strategies for mirror trading."""
        engine = MirrorTradingEngine()
        
        # Test various exit scenarios
        scenarios = [
            {
                "name": "institutional_exit",
                "position": {
                    "ticker": "XYZ",
                    "entry_price": 100,
                    "current_price": 120,
                    "institution_status": "exited"
                },
                "expected_action": "exit"
            },
            {
                "name": "target_reached",
                "position": {
                    "ticker": "ABC",
                    "entry_price": 50,
                    "current_price": 59,  # 18% gain (exceeds optimized 16.89% threshold)
                    "institution_status": "holding"
                },
                "expected_action": "reduce"
            },
            {
                "name": "stop_loss",
                "position": {
                    "ticker": "DEF",
                    "entry_price": 100,
                    "current_price": 75,  # 25% loss (exceeds optimized -23.29% threshold)
                    "institution_status": "holding"
                },
                "expected_action": "exit"
            }
        ]
        
        for scenario in scenarios:
            exit_decision = engine.determine_exit_strategy(scenario["position"])
            assert exit_decision["action"] == scenario["expected_action"]
            
    def test_mirror_correlation_analysis(self):
        """Test correlation analysis between our returns and institutional returns."""
        engine = MirrorTradingEngine()
        
        # Mock return data
        return_data = {
            "our_returns": [0.05, -0.02, 0.08, 0.03, -0.01, 0.12, 0.04],
            "institutional_returns": [0.06, -0.015, 0.075, 0.035, -0.005, 0.11, 0.045],
            "market_returns": [0.04, -0.01, 0.06, 0.02, 0.005, 0.09, 0.03]
        }
        
        correlation_analysis = engine.analyze_return_correlation(return_data)
        
        assert correlation_analysis["our_vs_institutional"] > 0.8  # High correlation
        assert correlation_analysis["our_vs_market"] > 0.7  # Decent market correlation
        assert correlation_analysis["tracking_error"] < 0.02  # Low tracking error
        assert correlation_analysis["mirror_effectiveness"] > 0.8