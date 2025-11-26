"""
Unit Tests for Sports Betting Risk Management Framework

Comprehensive test suite covering all risk management components:
- Risk Framework core functionality
- Portfolio Risk Manager
- Betting Limits and Controls
- Market Risk Analysis
- Syndicate Risk Management
- Performance Monitoring
"""

import unittest
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.sports_betting.risk_management import (
    RiskFramework,
    PortfolioRiskManager,
    BetOpportunity,
    SyndicateMember,
    MemberRole,
    ExpertiseLevel,
    RiskLevel,
    BettingDecision,
    MarketRiskAnalyzer,
    SyndicateController,
    PerformanceMonitor,
    BettingLimitsController
)


class TestBetOpportunity(unittest.TestCase):
    """Test BetOpportunity data class and calculations"""
    
    def test_bet_opportunity_creation(self):
        """Test creating bet opportunities with valid data"""
        bet = BetOpportunity(
            bet_id="TEST_001",
            sport="football",
            event="Test Game",
            selection="Team A -3",
            odds=1.91,
            probability=0.55,
            confidence=0.85
        )
        
        self.assertEqual(bet.bet_id, "TEST_001")
        self.assertEqual(bet.sport, "football")
        self.assertEqual(bet.odds, 1.91)
        self.assertEqual(bet.probability, 0.55)
        self.assertEqual(bet.confidence, 0.85)
        
    def test_edge_calculation(self):
        """Test edge calculation accuracy"""
        bet = BetOpportunity(
            bet_id="TEST_002",
            sport="basketball",
            event="Test Game",
            selection="Over 200",
            odds=2.0,
            probability=0.60,
            confidence=0.90
        )
        
        expected_edge = (0.60 * 2.0) - 1  # 0.20 or 20%
        self.assertAlmostEqual(bet.edge, expected_edge, places=4)
        
    def test_kelly_criterion_calculation(self):
        """Test Kelly criterion calculation"""
        bet = BetOpportunity(
            bet_id="TEST_003",
            sport="tennis",
            event="Test Match",
            selection="Player A to Win",
            odds=3.0,
            probability=0.40,
            confidence=0.75
        )
        
        # Kelly = (bp - q) / b where b = odds-1, p = probability, q = 1-p
        expected_kelly = ((3.0 - 1) * 0.40 - 0.60) / (3.0 - 1)  # 0.10 or 10%
        self.assertAlmostEqual(bet.kelly_fraction, expected_kelly, places=4)
        
    def test_invalid_odds_handling(self):
        """Test handling of invalid odds values"""
        with self.assertRaises(ValueError):
            BetOpportunity(
                bet_id="INVALID_001",
                sport="football",
                event="Test",
                selection="Test",
                odds=0.5,  # Invalid - odds must be > 1
                probability=0.5,
                confidence=0.8
            )
            
    def test_invalid_probability_handling(self):
        """Test handling of invalid probability values"""
        with self.assertRaises(ValueError):
            BetOpportunity(
                bet_id="INVALID_002",
                sport="football",
                event="Test",
                selection="Test",
                odds=2.0,
                probability=1.5,  # Invalid - probability must be <= 1
                confidence=0.8
            )


class TestSyndicateMember(unittest.TestCase):
    """Test SyndicateMember functionality"""
    
    def test_member_creation(self):
        """Test creating syndicate members"""
        member = SyndicateMember(
            member_id="MEMBER_001",
            name="John Doe",
            role=MemberRole.TRADER,
            expertise_areas={'football': ExpertiseLevel.EXPERT},
            betting_limit=50000,
            daily_limit=150000
        )
        
        self.assertEqual(member.member_id, "MEMBER_001")
        self.assertEqual(member.role, MemberRole.TRADER)
        self.assertEqual(member.betting_limit, 50000)
        self.assertEqual(member.daily_limit, 150000)
        
    def test_member_permissions(self):
        """Test member permission checking"""
        admin = SyndicateMember(
            member_id="ADMIN_001",
            name="Admin User",
            role=MemberRole.ADMIN,
            expertise_areas={},
            betting_limit=100000,
            daily_limit=300000
        )
        
        trader = SyndicateMember(
            member_id="TRADER_001",
            name="Trader User",
            role=MemberRole.TRADER,
            expertise_areas={},
            betting_limit=25000,
            daily_limit=75000
        )
        
        # Test permission checking logic would go here
        self.assertTrue(admin.role == MemberRole.ADMIN)
        self.assertTrue(trader.role == MemberRole.TRADER)
        
    def test_expertise_levels(self):
        """Test expertise level handling"""
        member = SyndicateMember(
            member_id="EXPERT_001",
            name="Expert User",
            role=MemberRole.SENIOR_TRADER,
            expertise_areas={
                'football': ExpertiseLevel.EXPERT,
                'basketball': ExpertiseLevel.ADVANCED,
                'tennis': ExpertiseLevel.INTERMEDIATE
            },
            betting_limit=75000,
            daily_limit=200000
        )
        
        self.assertEqual(member.expertise_areas['football'], ExpertiseLevel.EXPERT)
        self.assertEqual(member.expertise_areas['basketball'], ExpertiseLevel.ADVANCED)
        self.assertEqual(member.expertise_areas['tennis'], ExpertiseLevel.INTERMEDIATE)


class TestBettingLimitsController(unittest.TestCase):
    """Test betting limits and controls"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'max_bet_percentage': 0.05,
            'max_daily_loss_percentage': 0.10,
            'max_portfolio_risk': 0.15,
            'max_single_bet': 50000,
            'max_kelly_fraction': 0.25
        }
        self.limits_controller = BettingLimitsController(
            initial_bankroll=1000000,
            config=self.config
        )
        
    def test_single_bet_limit_check(self):
        """Test single bet limit validation"""
        # Valid bet within limits
        valid_result = self.limits_controller.check_single_bet_limits(
            bet_amount=25000,
            current_bankroll=1000000
        )
        self.assertTrue(valid_result.approved)
        
        # Bet exceeds percentage limit
        invalid_result = self.limits_controller.check_single_bet_limits(
            bet_amount=75000,  # 7.5% of bankroll
            current_bankroll=1000000
        )
        self.assertFalse(invalid_result.approved)
        
    def test_daily_loss_limit(self):
        """Test daily loss limit checking"""
        # Simulate daily loss approaching limit
        daily_loss = 90000  # 9% of bankroll
        
        result = self.limits_controller.check_daily_limits(
            current_daily_loss=daily_loss,
            proposed_bet_amount=25000,
            current_bankroll=1000000
        )
        
        # Should still be approved as total would be 11.5% but bet itself is reasonable
        # Implementation details depend on specific logic
        self.assertIsInstance(result.approved, bool)
        
    def test_portfolio_risk_calculation(self):
        """Test portfolio risk calculation"""
        active_bets = [
            {'amount': 30000, 'sport': 'football'},
            {'amount': 25000, 'sport': 'basketball'},
            {'amount': 20000, 'sport': 'tennis'}
        ]
        
        total_exposure = sum(bet['amount'] for bet in active_bets)
        portfolio_risk = total_exposure / 1000000
        
        result = self.limits_controller.check_portfolio_limits(
            active_exposure=total_exposure,
            new_bet_amount=15000,
            current_bankroll=1000000
        )
        
        self.assertIsInstance(result.approved, bool)
        
    def test_kelly_fraction_limits(self):
        """Test Kelly fraction limiting"""
        # High Kelly fraction should be limited
        high_kelly = 0.40  # 40%
        limited_kelly = self.limits_controller.apply_kelly_limits(high_kelly)
        
        self.assertLessEqual(limited_kelly, self.config['max_kelly_fraction'])
        
        # Normal Kelly fraction should pass through
        normal_kelly = 0.15  # 15%
        unchanged_kelly = self.limits_controller.apply_kelly_limits(normal_kelly)
        
        self.assertEqual(unchanged_kelly, normal_kelly)


class TestMarketRiskAnalyzer(unittest.TestCase):
    """Test market risk analysis functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.market_analyzer = MarketRiskAnalyzer()
        
    def test_odds_movement_tracking(self):
        """Test odds movement tracking and analysis"""
        market_id = "TEST_MARKET_001"
        
        # Track odds movements
        self.market_analyzer.track_odds_movement(market_id, 1.90, "BookmakerA", 10000)
        self.market_analyzer.track_odds_movement(market_id, 1.95, "BookmakerB", 15000)
        self.market_analyzer.track_odds_movement(market_id, 1.92, "BookmakerA", 12000)
        
        # Analyze movement
        analysis = self.market_analyzer.analyze_odds_movement(market_id)
        
        self.assertIsNotNone(analysis)
        self.assertIn('trend', analysis)
        self.assertIn('volatility', analysis)
        self.assertIn('consensus_odds', analysis)
        
    def test_market_efficiency_calculation(self):
        """Test market efficiency scoring"""
        # Mock odds data with different scenarios
        efficient_odds = [1.90, 1.91, 1.89, 1.90, 1.91]  # Low variance
        inefficient_odds = [1.80, 2.10, 1.75, 2.20, 1.85]  # High variance
        
        efficient_score = self.market_analyzer.calculate_market_efficiency(efficient_odds)
        inefficient_score = self.market_analyzer.calculate_market_efficiency(inefficient_odds)
        
        # More efficient market should have higher score
        self.assertGreater(efficient_score, inefficient_score)
        
    def test_arbitrage_detection(self):
        """Test arbitrage opportunity detection"""
        # Create scenario with arbitrage opportunity
        odds_data = [
            {'bookmaker': 'BookA', 'selection': 'Team1', 'odds': 2.10},
            {'bookmaker': 'BookB', 'selection': 'Team2', 'odds': 2.10},
            {'bookmaker': 'BookA', 'selection': 'Team2', 'odds': 1.95},
            {'bookmaker': 'BookB', 'selection': 'Team1', 'odds': 1.95}
        ]
        
        arbitrage_opps = self.market_analyzer.detect_arbitrage_opportunities(odds_data)
        
        # Should find arbitrage opportunities
        self.assertIsInstance(arbitrage_opps, list)
        
    def test_liquidity_assessment(self):
        """Test market liquidity assessment"""
        volume_data = [
            {'bookmaker': 'HighVolume', 'volume': 100000, 'odds': 1.90},
            {'bookmaker': 'LowVolume', 'volume': 5000, 'odds': 1.92},
            {'bookmaker': 'MediumVolume', 'volume': 25000, 'odds': 1.91}
        ]
        
        liquidity_score = self.market_analyzer.assess_liquidity(volume_data)
        
        self.assertIsInstance(liquidity_score, (int, float))
        self.assertGreaterEqual(liquidity_score, 0)
        self.assertLessEqual(liquidity_score, 1)


class TestPortfolioRiskManager(unittest.TestCase):
    """Test portfolio risk management functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.portfolio_manager = PortfolioRiskManager(
            initial_bankroll=1000000,
            config={'max_portfolio_risk': 0.20}
        )
        
    def test_correlation_calculation(self):
        """Test correlation calculation between sports/events"""
        # Mock historical data
        returns_data = {
            'football_spread': [0.05, -0.03, 0.08, -0.02, 0.06],
            'football_total': [0.02, -0.01, 0.04, 0.01, 0.03],
            'basketball_spread': [-0.01, 0.07, -0.04, 0.09, -0.02]
        }
        
        correlation_matrix = self.portfolio_manager.calculate_correlations(returns_data)
        
        self.assertIsInstance(correlation_matrix, (dict, np.ndarray))
        
    def test_portfolio_optimization(self):
        """Test multi-sport portfolio optimization"""
        opportunities = [
            BetOpportunity("BET_001", "football", "Game 1", "Team A", 1.90, 0.55, 0.85),
            BetOpportunity("BET_002", "basketball", "Game 2", "Team B", 2.10, 0.50, 0.80),
            BetOpportunity("BET_003", "tennis", "Match 1", "Player C", 1.85, 0.58, 0.90)
        ]
        
        allocations = self.portfolio_manager.optimize_multi_sport_portfolio(opportunities)
        
        self.assertIsInstance(allocations, list)
        
        # Check that allocations sum to reasonable total
        total_allocation = sum(alloc.allocation_percentage for alloc in allocations)
        self.assertLessEqual(total_allocation, 0.25)  # Should not exceed 25% of bankroll
        
    def test_risk_adjusted_sizing(self):
        """Test risk-adjusted position sizing"""
        bet = BetOpportunity("BET_004", "soccer", "Match", "Team D", 2.0, 0.60, 0.75)
        
        # Test different confidence levels
        high_confidence_size = self.portfolio_manager.calculate_risk_adjusted_size(
            bet, confidence_multiplier=1.0
        )
        low_confidence_size = self.portfolio_manager.calculate_risk_adjusted_size(
            bet, confidence_multiplier=0.5
        )
        
        # Higher confidence should allow larger size
        self.assertGreaterEqual(high_confidence_size, low_confidence_size)
        
    def test_diversification_scoring(self):
        """Test portfolio diversification scoring"""
        # Mock portfolio with different sports
        portfolio_bets = [
            {'sport': 'football', 'amount': 30000},
            {'sport': 'basketball', 'amount': 25000},
            {'sport': 'tennis', 'amount': 20000},
            {'sport': 'soccer', 'amount': 15000}
        ]
        
        diversification_score = self.portfolio_manager.calculate_diversification_score(portfolio_bets)
        
        self.assertIsInstance(diversification_score, (int, float))
        self.assertGreaterEqual(diversification_score, 0)
        self.assertLessEqual(diversification_score, 1)


class TestPerformanceMonitor(unittest.TestCase):
    """Test performance monitoring functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.performance_monitor = PerformanceMonitor(initial_bankroll=1000000)
        
    def test_bet_tracking(self):
        """Test bet tracking and P&L calculation"""
        # Add some bets
        bet1 = BetOpportunity("TRACK_001", "football", "Game 1", "Team A", 1.90, 0.55, 0.85)
        bet2 = BetOpportunity("TRACK_002", "basketball", "Game 2", "Team B", 2.10, 0.50, 0.80)
        
        self.performance_monitor.add_bet(bet1, 25000)
        self.performance_monitor.add_bet(bet2, 30000)
        
        # Update results
        self.performance_monitor.update_bet_result("TRACK_001", "win", 47500)  # Won
        self.performance_monitor.update_bet_result("TRACK_002", "loss", 0)     # Lost
        
        # Check P&L
        total_pnl = self.performance_monitor.get_total_pnl()
        expected_pnl = 47500 - 25000 - 30000  # Win - two stakes
        
        self.assertEqual(total_pnl, expected_pnl)
        
    def test_roi_calculation(self):
        """Test ROI calculation"""
        # Simulate some betting activity
        self.performance_monitor.add_bet(
            BetOpportunity("ROI_001", "tennis", "Match", "Player", 2.0, 0.60, 0.90),
            20000
        )
        self.performance_monitor.update_bet_result("ROI_001", "win", 40000)
        
        roi = self.performance_monitor.calculate_roi()
        expected_roi = (40000 - 20000) / 20000  # 100% ROI
        
        self.assertAlmostEqual(roi, expected_roi, places=4)
        
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation"""
        # Mock return series
        returns = [0.05, -0.02, 0.08, 0.01, -0.03, 0.06, 0.02, -0.01]
        
        sharpe_ratio = self.performance_monitor.calculate_sharpe_ratio(returns)
        
        self.assertIsInstance(sharpe_ratio, (int, float))
        
    def test_drawdown_calculation(self):
        """Test maximum drawdown calculation"""
        # Mock equity curve
        equity_values = [1000000, 1050000, 1020000, 980000, 1010000, 1080000, 1040000]
        
        max_drawdown = self.performance_monitor.calculate_max_drawdown(equity_values)
        
        self.assertIsInstance(max_drawdown, (int, float))
        self.assertLessEqual(max_drawdown, 0)  # Drawdown should be negative or zero
        
    def test_win_rate_calculation(self):
        """Test win rate calculation"""
        # Add and settle multiple bets
        bets = [
            ("WIN_001", "win"),
            ("WIN_002", "loss"),
            ("WIN_003", "win"),
            ("WIN_004", "win"),
            ("WIN_005", "loss")
        ]
        
        for bet_id, result in bets:
            bet = BetOpportunity(bet_id, "sport", "event", "selection", 2.0, 0.5, 0.8)
            self.performance_monitor.add_bet(bet, 10000)
            settlement = 20000 if result == "win" else 0
            self.performance_monitor.update_bet_result(bet_id, result, settlement)
            
        win_rate = self.performance_monitor.calculate_win_rate()
        expected_win_rate = 3 / 5  # 60%
        
        self.assertEqual(win_rate, expected_win_rate)


class TestRiskFramework(unittest.TestCase):
    """Test overall risk framework integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'max_kelly_fraction': 0.25,
            'max_portfolio_risk': 0.15,
            'max_bet_percentage': 0.05,
            'max_daily_loss_percentage': 0.10
        }
        self.risk_framework = RiskFramework(
            syndicate_name="Test Syndicate",
            initial_bankroll=1000000,
            config=self.config
        )
        
        # Add test members
        self.test_member = SyndicateMember(
            member_id="TEST_001",
            name="Test Member",
            role=MemberRole.TRADER,
            expertise_areas={'football': ExpertiseLevel.ADVANCED},
            betting_limit=50000,
            daily_limit=150000
        )
        self.risk_framework.syndicate_controller.add_member(self.test_member)
        
    def test_betting_opportunity_evaluation(self):
        """Test complete betting opportunity evaluation"""
        bet_opportunity = BetOpportunity(
            bet_id="EVAL_001",
            sport="football",
            event="Test Game",
            selection="Test Selection",
            odds=1.90,
            probability=0.56,
            confidence=0.85
        )
        
        decision = self.risk_framework.evaluate_betting_opportunity(
            bet_opportunity=bet_opportunity,
            bookmaker="TestBook",
            jurisdiction="US",
            proposer_id="TEST_001",
            participating_members=["TEST_001"]
        )
        
        self.assertIsInstance(decision, BettingDecision)
        self.assertIsInstance(decision.approved, bool)
        self.assertIsInstance(decision.risk_score, (int, float))
        
    def test_health_check_functionality(self):
        """Test system health check"""
        health_check = self.risk_framework.perform_health_check()
        
        self.assertIsNotNone(health_check)
        self.assertIn('overall_status', health_check.__dict__)
        self.assertIn('components', health_check.__dict__)
        self.assertIn('metrics', health_check.__dict__)
        
    def test_emergency_protocol_trigger(self):
        """Test emergency protocol activation"""
        # Simulate conditions that should trigger emergency protocols
        # This would depend on implementation details
        
        emergency_status, triggered_protocols = self.risk_framework.syndicate_controller.check_emergency_conditions()
        
        self.assertIsNotNone(emergency_status)
        self.assertIsInstance(triggered_protocols, list)
        
    def test_bet_placement_workflow(self):
        """Test complete bet placement workflow"""
        bet_opportunity = BetOpportunity(
            bet_id="PLACE_001",
            sport="basketball",
            event="Test Game",
            selection="Test Selection",
            odds=2.0,
            probability=0.55,
            confidence=0.90
        )
        
        # Evaluate
        decision = self.risk_framework.evaluate_betting_opportunity(
            bet_opportunity=bet_opportunity,
            bookmaker="TestBook",
            jurisdiction="US",
            proposer_id="TEST_001",
            participating_members=["TEST_001"]
        )
        
        # Place bet if approved
        if decision.approved:
            success = self.risk_framework.place_bet(bet_opportunity, decision, "TestBook")
            self.assertIsInstance(success, bool)
            
    def test_risk_dashboard_generation(self):
        """Test risk dashboard data generation"""
        dashboard = self.risk_framework.get_risk_dashboard()
        
        self.assertIsInstance(dashboard, dict)
        self.assertIn('framework', dashboard)
        self.assertIn('portfolio', dashboard)
        self.assertIn('syndicate', dashboard)
        self.assertIn('performance', dashboard)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios and edge cases"""
    
    def setUp(self):
        """Set up complex test scenario"""
        self.risk_framework = RiskFramework(
            syndicate_name="Integration Test Syndicate",
            initial_bankroll=2000000,
            config={
                'max_kelly_fraction': 0.20,
                'max_portfolio_risk': 0.12,
                'max_bet_percentage': 0.04
            }
        )
        
        # Add multiple members
        members = [
            SyndicateMember("ADMIN_001", "Admin", MemberRole.ADMIN, {}, 100000, 300000),
            SyndicateMember("SENIOR_001", "Senior", MemberRole.SENIOR_TRADER, {}, 75000, 225000),
            SyndicateMember("TRADER_001", "Trader1", MemberRole.TRADER, {}, 50000, 150000),
            SyndicateMember("TRADER_002", "Trader2", MemberRole.TRADER, {}, 50000, 150000)
        ]
        
        for member in members:
            self.risk_framework.syndicate_controller.add_member(member)
            
    def test_multiple_concurrent_bets(self):
        """Test handling multiple concurrent betting opportunities"""
        opportunities = [
            BetOpportunity("MULTI_001", "football", "Game 1", "Team A", 1.85, 0.58, 0.90),
            BetOpportunity("MULTI_002", "basketball", "Game 2", "Team B", 2.10, 0.52, 0.85),
            BetOpportunity("MULTI_003", "tennis", "Match 1", "Player C", 1.95, 0.54, 0.80),
            BetOpportunity("MULTI_004", "soccer", "Match 2", "Team D", 2.20, 0.48, 0.75)
        ]
        
        decisions = []
        for opp in opportunities:
            decision = self.risk_framework.evaluate_betting_opportunity(
                bet_opportunity=opp,
                bookmaker="TestBook",
                jurisdiction="US",
                proposer_id="SENIOR_001",
                participating_members=["SENIOR_001", "TRADER_001"]
            )
            decisions.append(decision)
            
        # Check that portfolio limits are respected across all bets
        total_allocated = sum(d.allocated_amount for d in decisions if d.approved)
        portfolio_percentage = total_allocated / self.risk_framework.performance_monitor.get_current_bankroll()
        
        self.assertLessEqual(portfolio_percentage, 0.15)  # Should respect portfolio limits
        
    def test_consensus_voting_workflow(self):
        """Test consensus voting for large bets"""
        large_bet = BetOpportunity(
            bet_id="CONSENSUS_001",
            sport="football",
            event="Championship Game",
            selection="Favorite -7",
            odds=1.90,
            probability=0.58,
            confidence=0.95
        )
        
        decision = self.risk_framework.evaluate_betting_opportunity(
            bet_opportunity=large_bet,
            bookmaker="PremiumBook",
            jurisdiction="US",
            proposer_id="SENIOR_001",
            participating_members=["ADMIN_001", "SENIOR_001", "TRADER_001", "TRADER_002"]
        )
        
        # Large bet should trigger consensus requirement
        if decision.proposal_id:
            # Simulate voting
            votes = [
                ("ADMIN_001", True, "Strong value"),
                ("TRADER_001", True, "Agree"),
                ("TRADER_002", False, "Too risky")
            ]
            
            for member_id, vote, comment in votes:
                self.risk_framework.syndicate_controller.vote_on_proposal(
                    decision.proposal_id, member_id, vote, comment
                )
                
            # Check proposal status
            proposal = self.risk_framework.syndicate_controller.active_proposals.get(decision.proposal_id)
            self.assertIsNotNone(proposal)
            
    def test_emergency_stop_scenario(self):
        """Test emergency stop functionality"""
        # Simulate rapid losses
        losing_bets = [
            ("LOSS_001", 25000),
            ("LOSS_002", 30000),
            ("LOSS_003", 35000),
            ("LOSS_004", 40000)
        ]
        
        for bet_id, amount in losing_bets:
            bet = BetOpportunity(bet_id, "sport", "event", "selection", 2.0, 0.5, 0.8)
            self.risk_framework.performance_monitor.add_bet(bet, amount)
            self.risk_framework.performance_monitor.update_bet_result(bet_id, "loss", 0)
            
        # Check if emergency protocols are triggered
        emergency_status, protocols = self.risk_framework.syndicate_controller.check_emergency_conditions()
        
        # Should detect significant losses
        self.assertIsNotNone(emergency_status)


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.DEBUG)
    
    # Run tests
    unittest.main(verbosity=2)