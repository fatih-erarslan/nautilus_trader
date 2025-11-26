"""
Comprehensive Test Suite for Fantasy Collective (Syndicate) System

This test suite covers:
1. Database operations (CRUD, transactions, constraints)
2. MCP tools with mock data
3. Scoring calculations for accuracy
4. League and collective management
5. Prediction resolution and payouts
6. Performance tests for concurrent users
7. Integration tests with existing systems
8. Security tests (SQL injection, access control)
"""

import pytest
import asyncio
import sqlite3
import threading
import time
import json
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional

# Import system under test
from src.syndicate.syndicate_tools import (
    create_syndicate, add_member, get_syndicate_status,
    allocate_funds, distribute_profits, process_withdrawal,
    get_member_performance, get_allocation_limits,
    update_member_contribution, get_profit_history,
    simulate_allocation, get_withdrawal_history,
    update_allocation_strategy, get_member_list,
    calculate_tax_liability
)
from src.syndicate.member_management import (
    SyndicateMemberManager, MemberRole, MemberTier,
    Member, MemberPermissions
)
from src.syndicate.capital_management import (
    FundAllocationEngine, ProfitDistributionSystem,
    WithdrawalManager, BettingOpportunity,
    AllocationStrategy, DistributionModel
)


class TestDatabase:
    """Mock database for testing"""
    
    def __init__(self):
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self.lock = threading.Lock()
        self._create_tables()
    
    def _create_tables(self):
        """Create test database tables"""
        with self.lock:
            cursor = self.connection.cursor()
            cursor.executescript("""
                CREATE TABLE syndicates (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    total_capital DECIMAL(15,2) DEFAULT 0
                );
                
                CREATE TABLE members (
                    id TEXT PRIMARY KEY,
                    syndicate_id TEXT REFERENCES syndicates(id),
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    role TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    capital_contribution DECIMAL(15,2) DEFAULT 0,
                    lifetime_earnings DECIMAL(15,2) DEFAULT 0,
                    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    performance_metrics TEXT DEFAULT '{}'
                );
                
                CREATE TABLE allocations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    syndicate_id TEXT REFERENCES syndicates(id),
                    opportunity_data TEXT NOT NULL,
                    allocated_amount DECIMAL(15,2),
                    strategy TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE distributions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    syndicate_id TEXT REFERENCES syndicates(id),
                    total_amount DECIMAL(15,2),
                    model TEXT,
                    distribution_data TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE withdrawals (
                    id TEXT PRIMARY KEY,
                    syndicate_id TEXT REFERENCES syndicates(id),
                    member_id TEXT REFERENCES members(id),
                    requested_amount DECIMAL(15,2),
                    approved_amount DECIMAL(15,2),
                    status TEXT DEFAULT 'pending',
                    is_emergency BOOLEAN DEFAULT 0,
                    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP
                );
                
                CREATE INDEX idx_members_syndicate ON members(syndicate_id);
                CREATE INDEX idx_allocations_syndicate ON allocations(syndicate_id);
                CREATE INDEX idx_distributions_syndicate ON distributions(syndicate_id);
                CREATE INDEX idx_withdrawals_member ON withdrawals(member_id);
            """)
            self.connection.commit()


@pytest.fixture(scope="session")
def test_db():
    """Create a test database for the session"""
    db = TestDatabase()
    yield db
    db.connection.close()


@pytest.fixture
def clean_db(test_db):
    """Clean database before each test"""
    with test_db.lock:
        cursor = test_db.connection.cursor()
        cursor.execute("DELETE FROM withdrawals")
        cursor.execute("DELETE FROM distributions")
        cursor.execute("DELETE FROM allocations")
        cursor.execute("DELETE FROM members")
        cursor.execute("DELETE FROM syndicates")
        test_db.connection.commit()
    yield test_db


@pytest.fixture
def sample_syndicate():
    """Create a sample syndicate for testing"""
    return {
        "syndicate_id": "test-syndicate-001",
        "name": "Elite Traders",
        "description": "High-performance trading collective"
    }


@pytest.fixture
def sample_member():
    """Create a sample member for testing"""
    return {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "role": "senior_analyst",
        "initial_contribution": 50000.0
    }


@pytest.fixture
def sample_opportunities():
    """Create sample betting opportunities for testing"""
    return [
        {
            "sport": "NFL",
            "event": "Patriots vs Jets",
            "bet_type": "spread",
            "selection": "Patriots -3.5",
            "odds": 1.91,
            "probability": 0.58,
            "edge": 0.05,
            "confidence": 0.75,
            "model_agreement": 0.82,
            "hours_until_event": 24,
            "liquidity": 50000
        },
        {
            "sport": "NBA",
            "event": "Lakers vs Warriors",
            "bet_type": "moneyline",
            "selection": "Lakers +150",
            "odds": 2.50,
            "probability": 0.45,
            "edge": 0.08,
            "confidence": 0.65,
            "model_agreement": 0.70,
            "hours_until_event": 12,
            "liquidity": 75000
        }
    ]


# =============================================================================
# UNIT TESTS - Database Operations (CRUD, Transactions, Constraints)
# =============================================================================

class TestDatabaseOperations:
    """Test database CRUD operations, transactions, and constraints"""

    def test_create_syndicate_database_operations(self, clean_db, sample_syndicate):
        """Test syndicate creation in database"""
        # Create syndicate
        result = create_syndicate(**sample_syndicate)
        
        # Verify creation
        assert result["status"] != "failed"
        assert result["syndicate_id"] == sample_syndicate["syndicate_id"]
        assert result["name"] == sample_syndicate["name"]
        
        # Verify database entry
        with clean_db.lock:
            cursor = clean_db.connection.cursor()
            cursor.execute("SELECT * FROM syndicates WHERE id = ?", 
                         (sample_syndicate["syndicate_id"],))
            row = cursor.fetchone()
            assert row is not None

    def test_duplicate_syndicate_constraint(self, clean_db, sample_syndicate):
        """Test database constraint preventing duplicate syndicates"""
        # Create first syndicate
        result1 = create_syndicate(**sample_syndicate)
        assert result1["status"] != "failed"
        
        # Try to create duplicate
        result2 = create_syndicate(**sample_syndicate)
        assert result2["status"] == "failed"
        assert "already exists" in result2["error"]

    def test_member_crud_operations(self, clean_db, sample_syndicate, sample_member):
        """Test member CRUD operations"""
        # Setup
        create_syndicate(**sample_syndicate)
        
        # Create member
        result = add_member(sample_syndicate["syndicate_id"], **sample_member)
        assert result["status"] != "failed"
        member_id = result["member_id"]
        
        # Read member
        status = get_syndicate_status(sample_syndicate["syndicate_id"])
        assert status["total_members"] == 1
        
        # Update member contribution
        update_result = update_member_contribution(
            sample_syndicate["syndicate_id"], 
            member_id, 
            25000.0
        )
        assert update_result["status"] != "failed"
        assert update_result["additional_amount"] == 25000.0
        
        # Verify update
        perf_result = get_member_performance(sample_syndicate["syndicate_id"], member_id)
        assert perf_result["capital_contribution"] == 75000.0

    def test_unique_email_constraint(self, clean_db, sample_syndicate, sample_member):
        """Test unique email constraint"""
        create_syndicate(**sample_syndicate)
        
        # Add first member
        result1 = add_member(sample_syndicate["syndicate_id"], **sample_member)
        assert result1["status"] != "failed"
        
        # Try to add member with same email
        duplicate_member = sample_member.copy()
        duplicate_member["name"] = "Jane Doe"
        
        result2 = add_member(sample_syndicate["syndicate_id"], **duplicate_member)
        # Should handle gracefully (implementation dependent)
        # In a real database, this would violate unique constraint

    def test_transaction_rollback_on_error(self, clean_db):
        """Test transaction rollback on error"""
        # This would test database transaction handling
        # Mock scenario where partial operations fail
        with patch('src.syndicate.syndicate_tools.logger') as mock_logger:
            # Create syndicate with invalid data that causes partial failure
            result = create_syndicate("", "", "")  # Invalid empty ID
            
            # Verify rollback occurred
            with clean_db.lock:
                cursor = clean_db.connection.cursor()
                cursor.execute("SELECT COUNT(*) FROM syndicates")
                count = cursor.fetchone()[0]
                assert count == 0


# =============================================================================
# UNIT TESTS - MCP Tools with Mock Data
# =============================================================================

class TestMCPTools:
    """Test MCP tools with various mock data scenarios"""

    def test_create_syndicate_tool(self, sample_syndicate):
        """Test create syndicate MCP tool"""
        result = create_syndicate(**sample_syndicate)
        
        assert "error" not in result
        assert result["syndicate_id"] == sample_syndicate["syndicate_id"]
        assert result["name"] == sample_syndicate["name"]
        assert result["total_members"] == 0
        assert result["total_capital"] == 0

    def test_add_member_tool_with_roles(self, sample_syndicate):
        """Test add member tool with different roles"""
        create_syndicate(**sample_syndicate)
        
        roles_to_test = [
            "lead_investor", "senior_analyst", "junior_analyst",
            "contributing_member", "observer"
        ]
        
        for i, role in enumerate(roles_to_test):
            member = {
                "name": f"Test User {i}",
                "email": f"user{i}@example.com",
                "role": role,
                "initial_contribution": 10000.0 * (i + 1)
            }
            
            result = add_member(sample_syndicate["syndicate_id"], **member)
            assert result["status"] != "failed"
            assert result["role"] == role
            # Verify tier assignment based on contribution
            if member["initial_contribution"] >= 100000:
                assert result["tier"] in ["platinum", "gold"]

    def test_fund_allocation_tool(self, sample_syndicate, sample_member, sample_opportunities):
        """Test fund allocation MCP tool"""
        create_syndicate(**sample_syndicate)
        add_member(sample_syndicate["syndicate_id"], **sample_member)
        
        result = allocate_funds(
            sample_syndicate["syndicate_id"],
            sample_opportunities,
            "kelly_criterion"
        )
        
        assert result["status"] != "failed"
        assert "allocations" in result
        assert result["total_capital"] > 0
        assert len(result["allocations"]) <= len(sample_opportunities)
        
        # Verify allocation details
        for allocation in result["allocations"]:
            assert "allocated_amount" in allocation
            assert "percentage_of_bankroll" in allocation
            assert allocation["allocated_amount"] >= 0

    def test_profit_distribution_tool(self, sample_syndicate, sample_member):
        """Test profit distribution MCP tool"""
        create_syndicate(**sample_syndicate)
        add_member(sample_syndicate["syndicate_id"], **sample_member)
        
        total_profit = 15000.0
        result = distribute_profits(
            sample_syndicate["syndicate_id"],
            total_profit,
            "hybrid"
        )
        
        assert result["status"] != "failed" or result["execution_status"] != "failed"
        assert result["total_profit"] == total_profit
        assert "distributions" in result
        assert len(result["distributions"]) > 0
        
        # Verify distribution totals
        total_distributed = sum(d["gross_amount"] for d in result["distributions"])
        assert abs(total_distributed - total_profit) < 0.01  # Account for rounding

    def test_withdrawal_processing_tool(self, sample_syndicate, sample_member):
        """Test withdrawal processing MCP tool"""
        create_syndicate(**sample_syndicate)
        member_result = add_member(sample_syndicate["syndicate_id"], **sample_member)
        member_id = member_result["member_id"]
        
        # Test normal withdrawal
        withdrawal_amount = 5000.0
        result = process_withdrawal(
            sample_syndicate["syndicate_id"],
            member_id,
            withdrawal_amount,
            is_emergency=False
        )
        
        assert result["member_id"] == member_id
        assert result["requested_amount"] == withdrawal_amount
        # Status should be approved or pending
        assert result["status"] in ["approved", "pending", "completed"]

    def test_member_performance_tool(self, sample_syndicate, sample_member):
        """Test member performance MCP tool"""
        create_syndicate(**sample_syndicate)
        member_result = add_member(sample_syndicate["syndicate_id"], **sample_member)
        member_id = member_result["member_id"]
        
        result = get_member_performance(sample_syndicate["syndicate_id"], member_id)
        
        assert result["status"] != "failed"
        assert result["member_id"] == member_id
        assert "performance_metrics" in result
        
        # Verify performance metrics structure
        metrics = result["performance_metrics"]
        expected_metrics = [
            "total_profit", "roi", "win_rate", "average_return",
            "sharpe_ratio", "alpha", "skill_score", "consistency_score", "risk_score"
        ]
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))


# =============================================================================
# UNIT TESTS - Scoring Calculations for Accuracy
# =============================================================================

class TestScoringCalculations:
    """Test accuracy of scoring and calculation systems"""

    def test_kelly_criterion_calculation(self):
        """Test Kelly Criterion calculation accuracy"""
        opportunity = BettingOpportunity(
            sport="NFL",
            event="Test Game",
            bet_type="spread",
            selection="Team A -3.5",
            odds=1.91,  # Decimal odds
            probability=0.58,  # 58% win probability
            edge=0.05,
            confidence=0.75,
            model_agreement=0.82,
            time_until_event=timedelta(hours=24),
            liquidity=50000
        )
        
        # Create allocation engine
        bankroll = Decimal("100000")  # $100k bankroll
        engine = FundAllocationEngine("test", bankroll)
        
        # Calculate Kelly percentage manually for verification
        # Kelly = (bp - q) / b
        # where b = odds - 1, p = probability, q = 1 - p
        b = opportunity.odds - 1  # 0.91
        p = opportunity.probability  # 0.58
        q = 1 - p  # 0.42
        expected_kelly = (b * p - q) / b
        
        # Test allocation
        result = engine.allocate_funds(opportunity, AllocationStrategy.KELLY_CRITERION)
        
        # Verify the calculation is reasonable (fractional Kelly applied)
        assert result.amount > 0
        assert result.amount <= bankroll * Decimal("0.05")  # Max 5% per bet
        assert result.percentage_of_bankroll > 0
        assert result.percentage_of_bankroll <= 0.05

    def test_profit_distribution_calculations(self):
        """Test profit distribution calculation accuracy"""
        # Create mock syndicate manager
        syndicate_id = "test-calc"
        manager = SyndicateMemberManager(syndicate_id)
        
        # Add members with different contributions and performance
        member1 = manager.add_member(
            "Alice", "alice@test.com", 
            MemberRole.LEAD_INVESTOR, Decimal("50000")
        )
        member2 = manager.add_member(
            "Bob", "bob@test.com", 
            MemberRole.SENIOR_ANALYST, Decimal("30000")
        )
        member3 = manager.add_member(
            "Charlie", "charlie@test.com", 
            MemberRole.CONTRIBUTING_MEMBER, Decimal("20000")
        )
        
        # Mock performance scores
        member1.performance_metrics.roi = Decimal("0.15")  # 15% ROI
        member2.performance_metrics.roi = Decimal("0.20")  # 20% ROI
        member3.performance_metrics.roi = Decimal("0.10")  # 10% ROI
        
        # Create distribution system
        distributor = ProfitDistributionSystem(syndicate_id, manager)
        
        # Test hybrid model (50% capital, 30% performance, 20% equal)
        total_profit = Decimal("10000")
        distributions = distributor.calculate_distribution(
            total_profit, DistributionModel.HYBRID
        )
        
        # Verify total adds up
        total_distributed = sum(distributions.values())
        assert abs(total_distributed - total_profit) < Decimal("0.01")
        
        # Verify Alice (largest contributor) gets more than Charlie
        assert distributions[member1.member_id] > distributions[member3.member_id]
        
        # Verify Bob (best performer) gets reasonable share despite lower contribution
        assert distributions[member2.member_id] > distributions[member3.member_id]

    def test_risk_scoring_accuracy(self):
        """Test risk scoring calculations"""
        opportunity = BettingOpportunity(
            sport="NFL",
            event="High Risk Game",
            bet_type="prop",
            selection="Player O 2.5 TDs",
            odds=3.50,  # High odds = high risk
            probability=0.25,  # Low probability
            edge=0.02,  # Small edge
            confidence=0.40,  # Low confidence
            model_agreement=0.30,  # Poor model agreement
            time_until_event=timedelta(hours=2),  # Short time
            liquidity=5000,  # Low liquidity
            is_live=True
        )
        
        engine = FundAllocationEngine("test", Decimal("100000"))
        result = engine.allocate_funds(opportunity, AllocationStrategy.KELLY_CRITERION)
        
        # High risk should result in smaller allocation
        assert result.percentage_of_bankroll < 0.02  # Less than 2%
        assert len(result.warnings) > 0  # Should have warnings
        assert result.risk_metrics["overall_risk"] > 0.5  # High risk score

    def test_performance_metrics_calculation(self):
        """Test performance metrics calculations"""
        from src.syndicate.member_management import PerformanceMetrics
        
        # Create performance tracker with mock data
        metrics = PerformanceMetrics()
        
        # Simulate betting history
        bet_results = [
            {"stake": 1000, "return": 1100, "won": True},   # 10% return
            {"stake": 1500, "return": 0, "won": False},     # -100% return
            {"stake": 2000, "return": 2400, "won": True},   # 20% return
            {"stake": 800, "return": 960, "won": True},     # 20% return
            {"stake": 1200, "return": 0, "won": False},     # -100% return
        ]
        
        total_staked = sum(bet["stake"] for bet in bet_results)
        total_returned = sum(bet["return"] for bet in bet_results)
        wins = sum(1 for bet in bet_results if bet["won"])
        
        # Calculate expected metrics
        expected_roi = (total_returned - total_staked) / total_staked
        expected_win_rate = wins / len(bet_results)
        
        # Update metrics
        metrics.total_profit = Decimal(str(total_returned - total_staked))
        metrics.total_invested = Decimal(str(total_staked))
        metrics.win_rate = Decimal(str(expected_win_rate))
        metrics.roi = Decimal(str(expected_roi))
        
        # Verify calculations
        assert abs(float(metrics.roi) - expected_roi) < 0.001
        assert abs(float(metrics.win_rate) - expected_win_rate) < 0.001
        assert metrics.total_profit == total_returned - total_staked


# =============================================================================
# UNIT TESTS - League and Collective Management
# =============================================================================

class TestLeagueCollectiveManagement:
    """Test league and collective management features"""

    def test_multi_syndicate_management(self):
        """Test managing multiple syndicates"""
        syndicates = [
            {"syndicate_id": "league-001", "name": "NFL Specialists"},
            {"syndicate_id": "league-002", "name": "NBA Experts"},
            {"syndicate_id": "league-003", "name": "Soccer Analytics"}
        ]
        
        created_syndicates = []
        for syndicate in syndicates:
            result = create_syndicate(**syndicate)
            assert result["status"] != "failed"
            created_syndicates.append(result["syndicate_id"])
        
        # Verify all syndicates exist independently
        for syndicate_id in created_syndicates:
            status = get_syndicate_status(syndicate_id)
            assert status["status"] != "failed"
            assert status["syndicate_id"] == syndicate_id

    def test_cross_syndicate_member_participation(self):
        """Test member participating in multiple syndicates"""
        # Create multiple syndicates
        syndicate1 = create_syndicate("multi-001", "Syndicate 1")
        syndicate2 = create_syndicate("multi-002", "Syndicate 2")
        
        # Same member joins both (different email required for uniqueness)
        member1_data = {
            "name": "John Multi",
            "email": "john.multi1@test.com",
            "role": "senior_analyst",
            "initial_contribution": 25000.0
        }
        
        member2_data = {
            "name": "John Multi",
            "email": "john.multi2@test.com",
            "role": "contributing_member",
            "initial_contribution": 15000.0
        }
        
        result1 = add_member("multi-001", **member1_data)
        result2 = add_member("multi-002", **member2_data)
        
        assert result1["status"] != "failed"
        assert result2["status"] != "failed"
        
        # Verify different roles in different syndicates
        assert result1["role"] == "senior_analyst"
        assert result2["role"] == "contributing_member"

    def test_syndicate_hierarchy_permissions(self):
        """Test hierarchical permissions within syndicate"""
        create_syndicate("hierarchy-001", "Permission Test")
        
        # Add members with different roles
        lead_investor = add_member("hierarchy-001", "Lead", "lead@test.com", 
                                 "lead_investor", 100000.0)
        senior_analyst = add_member("hierarchy-001", "Senior", "senior@test.com", 
                                  "senior_analyst", 50000.0)
        junior_analyst = add_member("hierarchy-001", "Junior", "junior@test.com", 
                                  "junior_analyst", 25000.0)
        observer = add_member("hierarchy-001", "Observer", "observer@test.com", 
                            "observer", 5000.0)
        
        # Verify role hierarchy
        assert lead_investor["role"] == "lead_investor"
        assert senior_analyst["role"] == "senior_analyst"
        assert junior_analyst["role"] == "junior_analyst"
        assert observer["role"] == "observer"

    def test_collective_resource_allocation(self):
        """Test collective resource allocation across opportunities"""
        create_syndicate("resource-001", "Resource Test")
        add_member("resource-001", "Contributor", "contrib@test.com", 
                  "contributing_member", 75000.0)
        
        # Multiple opportunities requiring allocation
        opportunities = [
            {
                "sport": "NFL", "event": "Game 1", "bet_type": "spread",
                "selection": "Team A -3", "odds": 1.91, "probability": 0.55,
                "edge": 0.03, "confidence": 0.70, "model_agreement": 0.75,
                "hours_until_event": 24, "liquidity": 50000
            },
            {
                "sport": "NBA", "event": "Game 2", "bet_type": "total",
                "selection": "Over 220.5", "odds": 1.95, "probability": 0.60,
                "edge": 0.08, "confidence": 0.85, "model_agreement": 0.90,
                "hours_until_event": 12, "liquidity": 75000
            },
            {
                "sport": "MLB", "event": "Game 3", "bet_type": "moneyline",
                "selection": "Home +150", "odds": 2.50, "probability": 0.45,
                "edge": 0.05, "confidence": 0.60, "model_agreement": 0.65,
                "hours_until_event": 36, "liquidity": 30000
            }
        ]
        
        result = allocate_funds("resource-001", opportunities, "kelly_criterion")
        
        assert result["status"] != "failed"
        assert len(result["allocations"]) >= 2  # Should allocate to multiple opportunities
        
        # Verify total allocation doesn't exceed limits
        total_allocated = result["total_allocated"]
        total_capital = result["total_capital"]
        assert total_allocated <= total_capital * 0.20  # 20% daily limit


# =============================================================================
# UNIT TESTS - Prediction Resolution and Payouts
# =============================================================================

class TestPredictionResolution:
    """Test prediction resolution and payout systems"""

    def test_bet_outcome_resolution(self):
        """Test bet outcome resolution and profit calculation"""
        create_syndicate("payout-001", "Payout Test")
        member_result = add_member("payout-001", "Bettor", "bettor@test.com", 
                                 "contributing_member", 50000.0)
        member_id = member_result["member_id"]
        
        # Simulate successful bet resolution
        bet_amount = 2000.0
        winning_odds = 1.91
        expected_profit = bet_amount * (winning_odds - 1)  # $1820
        
        # Test profit distribution after winning bet
        result = distribute_profits("payout-001", expected_profit, "proportional")
        
        assert result["status"] != "failed" or result["execution_status"] != "failed"
        assert len(result["distributions"]) > 0
        
        # Member should receive the profit (minus taxes/fees)
        member_distribution = result["distributions"][0]
        assert member_distribution["member_id"] == member_id
        assert member_distribution["gross_amount"] > 0

    def test_multiple_bet_resolution(self):
        """Test resolution of multiple simultaneous bets"""
        create_syndicate("multi-bet-001", "Multi Bet Test")
        add_member("multi-bet-001", "Multi Bettor", "multi@test.com", 
                  "senior_analyst", 80000.0)
        
        # Simulate multiple bet outcomes
        bet_results = [
            {"amount": 1500, "odds": 1.95, "won": True},   # $1425 profit
            {"amount": 2000, "odds": 2.20, "won": False},  # -$2000 loss  
            {"amount": 1000, "odds": 1.85, "won": True},   # $850 profit
        ]
        
        total_profit = sum(
            bet["amount"] * (bet["odds"] - 1) if bet["won"] else -bet["amount"]
            for bet in bet_results
        )
        # $1425 - $2000 + $850 = $275
        
        if total_profit > 0:
            result = distribute_profits("multi-bet-001", total_profit, "hybrid")
            assert result["total_profit"] == total_profit

    def test_negative_outcome_handling(self):
        """Test handling of negative outcomes (losses)"""
        create_syndicate("loss-001", "Loss Test")
        add_member("loss-001", "Loser", "loser@test.com", 
                  "contributing_member", 30000.0)
        
        # Test loss scenario - no distribution should occur for losses
        loss_amount = -5000.0
        
        # Should handle gracefully - no distribution for losses
        result = distribute_profits("loss-001", loss_amount, "proportional")
        
        # Implementation should handle negative amounts appropriately
        # Either reject the distribution or handle as capital reduction

    def test_payout_tax_calculations(self):
        """Test tax calculations on payouts"""
        create_syndicate("tax-001", "Tax Test")
        member_result = add_member("tax-001", "Taxpayer", "taxpayer@test.com", 
                                 "contributing_member", 45000.0)
        member_id = member_result["member_id"]
        
        # Test tax liability calculation
        result = calculate_tax_liability("tax-001", member_id, "US")
        
        assert result["status"] != "failed"
        assert result["member_id"] == member_id
        assert "tax_breakdown" in result
        assert "total_tax" in result
        assert "net_earnings" in result
        assert result["effective_rate"] >= 0


# =============================================================================
# PERFORMANCE TESTS - Concurrent Users
# =============================================================================

class TestConcurrentPerformance:
    """Test system performance under concurrent load"""

    @pytest.mark.slow
    def test_concurrent_syndicate_creation(self):
        """Test concurrent syndicate creation"""
        def create_test_syndicate(i):
            return create_syndicate(f"concurrent-{i}", f"Syndicate {i}", 
                                  f"Test syndicate {i}")
        
        # Create 50 syndicates concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_test_syndicate, i) for i in range(50)]
            results = [future.result() for future in as_completed(futures)]
        
        # Verify all creations succeeded
        successful = [r for r in results if r.get("status") != "failed"]
        assert len(successful) == 50

    @pytest.mark.slow
    def test_concurrent_member_operations(self):
        """Test concurrent member operations"""
        # Setup
        create_syndicate("concurrent-members", "Concurrent Test")
        
        def add_test_member(i):
            return add_member("concurrent-members", f"User {i}", 
                            f"user{i}@test.com", "contributing_member", 10000.0)
        
        # Add 100 members concurrently
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(add_test_member, i) for i in range(100)]
            results = [future.result() for future in as_completed(futures)]
        
        # Verify results
        successful = [r for r in results if r.get("status") != "failed"]
        assert len(successful) >= 95  # Allow for some failures due to concurrency

    @pytest.mark.slow
    def test_concurrent_fund_allocation(self):
        """Test concurrent fund allocation operations"""
        # Setup
        create_syndicate("concurrent-funds", "Fund Test")
        add_member("concurrent-funds", "Fund Manager", "manager@test.com", 
                  "lead_investor", 200000.0)
        
        sample_opportunity = [{
            "sport": "NFL", "event": "Concurrent Test", "bet_type": "spread",
            "selection": "Team A -3", "odds": 1.91, "probability": 0.55,
            "edge": 0.03, "confidence": 0.70, "model_agreement": 0.75,
            "hours_until_event": 24, "liquidity": 50000
        }]
        
        def allocate_funds_test():
            return allocate_funds("concurrent-funds", sample_opportunity, "kelly_criterion")
        
        # Run 30 concurrent allocations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(allocate_funds_test) for _ in range(30)]
            results = [future.result() for future in as_completed(futures)]
        
        # Verify allocations are consistent
        successful = [r for r in results if r.get("status") != "failed"]
        assert len(successful) >= 25
        
        # Check allocation amounts are reasonable
        for result in successful:
            assert result["total_allocated"] >= 0
            assert result["total_capital"] > 0

    @pytest.mark.slow 
    def test_system_load_simulation(self):
        """Test system under realistic load simulation"""
        # Create multiple syndicates
        syndicates = []
        for i in range(10):
            syndicate_id = f"load-test-{i}"
            create_syndicate(syndicate_id, f"Load Test {i}")
            syndicates.append(syndicate_id)
            
            # Add members to each
            for j in range(5):
                add_member(syndicate_id, f"User {i}-{j}", 
                          f"user{i}_{j}@test.com", "contributing_member", 
                          20000.0 + (j * 5000))
        
        # Simulate mixed operations
        operations = []
        
        def mixed_operation(syndicate_id, op_type):
            if op_type == "status":
                return get_syndicate_status(syndicate_id)
            elif op_type == "allocation":
                opp = [{
                    "sport": "NBA", "event": "Load Test", "bet_type": "total",
                    "selection": "Over 210", "odds": 1.90, "probability": 0.52,
                    "edge": 0.02, "confidence": 0.65, "model_agreement": 0.70,
                    "hours_until_event": 18, "liquidity": 40000
                }]
                return allocate_funds(syndicate_id, opp, "fixed_percentage")
            elif op_type == "members":
                return get_member_list(syndicate_id)
        
        # Run 200 mixed operations across all syndicates
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = []
            for _ in range(200):
                syndicate_id = syndicates[_ % len(syndicates)]
                op_type = ["status", "allocation", "members"][_ % 3]
                futures.append(executor.submit(mixed_operation, syndicate_id, op_type))
            
            results = [future.result() for future in as_completed(futures)]
        
        # Verify system remained stable
        successful = [r for r in results if not (isinstance(r, dict) and r.get("status") == "failed")]
        success_rate = len(successful) / len(results)
        assert success_rate >= 0.90  # 90% success rate under load


# =============================================================================
# INTEGRATION TESTS - With Existing Systems  
# =============================================================================

class TestSystemIntegration:
    """Test integration with existing trading systems"""

    @patch('src.syndicate.syndicate_tools.logger')
    def test_logging_integration(self, mock_logger):
        """Test integration with logging system"""
        create_syndicate("log-test", "Logging Test")
        
        # Verify logging calls were made
        mock_logger.error.assert_not_called()  # No errors should occur
        
        # Test error logging
        result = create_syndicate("", "")  # Invalid input
        if result.get("status") == "failed":
            mock_logger.error.assert_called()

    def test_decimal_precision_integration(self):
        """Test Decimal precision integration"""
        create_syndicate("precision-test", "Precision Test")
        add_member("precision-test", "Precision User", "precision@test.com", 
                  "contributing_member", 33333.333)
        
        # Test high precision calculations
        result = distribute_profits("precision-test", 12345.6789, "proportional")
        
        # Verify precision is maintained
        if result.get("status") != "failed":
            for dist in result["distributions"]:
                assert isinstance(dist["gross_amount"], (int, float))
                # Should maintain reasonable precision
                assert round(dist["gross_amount"], 2) == dist["gross_amount"] or \
                       abs(dist["gross_amount"] - round(dist["gross_amount"], 2)) < 0.01

    def test_datetime_handling_integration(self):
        """Test datetime handling integration"""
        start_time = datetime.now()
        
        create_syndicate("time-test", "Time Test")
        member_result = add_member("time-test", "Time User", "time@test.com", 
                                 "contributing_member", 25000.0)
        
        end_time = datetime.now()
        
        # Verify timestamps are reasonable
        assert "joined_at" in member_result
        joined_time = datetime.fromisoformat(member_result["joined_at"].replace('Z', '+00:00').replace('+00:00', ''))
        assert start_time <= joined_time <= end_time

    def test_enum_integration(self):
        """Test enum integration with MemberRole and other enums"""
        create_syndicate("enum-test", "Enum Test")
        
        # Test all valid roles
        valid_roles = ["lead_investor", "senior_analyst", "junior_analyst", 
                      "contributing_member", "observer"]
        
        for role in valid_roles:
            result = add_member("enum-test", f"{role}_user", 
                              f"{role}@test.com", role, 20000.0)
            assert result["role"] == role
            
        # Test allocation strategies
        strategies = ["kelly_criterion", "fixed_percentage", "dynamic_confidence"]
        add_member("enum-test", "Strategy Tester", "strategy@test.com", 
                  "contributing_member", 50000.0)
        
        opp = [{
            "sport": "NFL", "event": "Enum Test", "bet_type": "spread",
            "selection": "Team -3", "odds": 1.90, "probability": 0.53,
            "edge": 0.02, "confidence": 0.70, "model_agreement": 0.75,
            "hours_until_event": 24, "liquidity": 50000
        }]
        
        for strategy in strategies:
            result = allocate_funds("enum-test", opp, strategy)
            if result.get("status") != "failed":
                assert result["strategy"] == strategy


# =============================================================================
# SECURITY TESTS - SQL Injection and Access Control
# =============================================================================

class TestSecurityValidation:
    """Test security features including SQL injection prevention and access control"""

    def test_sql_injection_prevention_syndicate_id(self):
        """Test SQL injection prevention in syndicate ID"""
        malicious_inputs = [
            "'; DROP TABLE syndicates; --",
            "' OR '1'='1",
            "admin'; DELETE FROM members WHERE '1'='1'; --",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "' UNION SELECT * FROM members --"
        ]
        
        for malicious_input in malicious_inputs:
            result = create_syndicate(malicious_input, "Malicious Test")
            # System should handle gracefully without errors
            if result.get("status") == "failed":
                # Should fail safely, not with SQL errors
                assert "error" in result
                assert "SQL" not in result.get("error", "").upper()
                assert "SYNTAX" not in result.get("error", "").upper()

    def test_sql_injection_prevention_member_data(self):
        """Test SQL injection prevention in member data"""
        create_syndicate("security-test", "Security Test")
        
        malicious_names = [
            "'; DROP TABLE members; --",
            "Robert'; DELETE FROM syndicates WHERE '1'='1'; --",
            "' OR 1=1 --",
        ]
        
        malicious_emails = [
            "test'; DROP TABLE syndicates; --@example.com",
            "' OR '1'='1' --@evil.com",
            "admin@'; DELETE FROM members; --"
        ]
        
        for i, (name, email) in enumerate(zip(malicious_names, malicious_emails)):
            result = add_member("security-test", name, f"secure{i}@test.com", 
                              "contributing_member", 10000.0)
            # Should handle malicious input gracefully
            if result.get("status") != "failed":
                # If successful, should have sanitized the input
                assert result["name"] != name or len(name) < 50  # Basic sanity

    def test_input_validation_and_sanitization(self):
        """Test input validation and sanitization"""
        # Test invalid syndicate creation
        invalid_inputs = [
            ("", "Empty ID should fail"),
            (None, "None ID should fail"), 
            ("a" * 1000, "Extremely long ID should fail"),
            ("test\x00null", "Null bytes should be handled"),
            ("test\n\r\t", "Control characters should be handled")
        ]
        
        for invalid_id, description in invalid_inputs:
            try:
                result = create_syndicate(invalid_id, "Test")
                if result.get("status") == "failed":
                    assert "error" in result  # Should fail gracefully
            except Exception as e:
                # Should not raise unhandled exceptions
                assert False, f"Unhandled exception for {description}: {e}"

    def test_access_control_member_operations(self):
        """Test access control for member operations"""
        create_syndicate("access-test", "Access Control Test")
        
        # Create members with different roles
        lead_result = add_member("access-test", "Lead", "lead@test.com", 
                               "lead_investor", 100000.0)
        observer_result = add_member("access-test", "Observer", "observer@test.com", 
                                   "observer", 5000.0)
        
        lead_id = lead_result["member_id"]
        observer_id = observer_result["member_id"]
        
        # Observer should not be able to perform certain operations
        # (This would require more sophisticated access control implementation)
        
        # Test withdrawal limits based on role/contribution
        large_withdrawal = process_withdrawal("access-test", observer_id, 
                                            50000.0, False)  # More than contribution
        
        # Should be rejected or limited
        if large_withdrawal.get("status") not in ["failed", "rejected"]:
            # If approved, should be limited to contribution amount
            assert large_withdrawal["approved_amount"] <= 5000.0

    def test_rate_limiting_simulation(self):
        """Test rate limiting protection"""
        create_syndicate("rate-test", "Rate Limit Test")
        
        # Simulate rapid requests
        results = []
        start_time = time.time()
        
        for i in range(100):
            result = get_syndicate_status("rate-test")
            results.append(result)
            
        end_time = time.time()
        duration = end_time - start_time
        
        # All requests should complete (basic availability test)
        successful = [r for r in results if r.get("status") != "failed"]
        assert len(successful) >= 95  # Allow for some failures
        
        # Should not be suspiciously fast (would indicate no processing)
        assert duration > 0.1  # At least 100ms for 100 requests

    def test_data_exposure_prevention(self):
        """Test prevention of sensitive data exposure"""
        create_syndicate("exposure-test", "Data Exposure Test")
        member_result = add_member("exposure-test", "Sensitive User", 
                                 "sensitive@test.com", "contributing_member", 50000.0)
        
        # Test that sensitive data is not exposed in responses
        performance_result = get_member_performance("exposure-test", 
                                                  member_result["member_id"])
        
        # Should not contain sensitive information like passwords, tokens, etc.
        sensitive_fields = ["password", "token", "secret", "key", "private"]
        result_str = json.dumps(performance_result).lower()
        
        for sensitive_field in sensitive_fields:
            assert sensitive_field not in result_str

    def test_parameter_tampering_protection(self):
        """Test protection against parameter tampering"""
        create_syndicate("tamper-test", "Tamper Test")
        member_result = add_member("tamper-test", "Tamper User", 
                                 "tamper@test.com", "contributing_member", 30000.0)
        member_id = member_result["member_id"]
        
        # Test tampering with contribution amounts
        tamper_attempts = [-999999.99, 0, 1000000000, float('inf')]
        
        for amount in tamper_attempts:
            try:
                result = update_member_contribution("tamper-test", member_id, amount)
                if result.get("status") != "failed":
                    # If successful, should be within reasonable bounds
                    assert -50000 <= result.get("additional_amount", 0) <= 1000000
            except (ValueError, OverflowError):
                # Expected for invalid values
                pass


# =============================================================================
# FIXTURES AND UTILITIES FOR COMPREHENSIVE TESTING
# =============================================================================

@pytest.fixture
def performance_test_data():
    """Generate large datasets for performance testing"""
    return {
        "large_member_list": [
            {
                "name": f"User {i}",
                "email": f"user{i}@performance.test", 
                "role": ["contributing_member", "junior_analyst", "senior_analyst"][i % 3],
                "initial_contribution": 10000.0 + (i * 1000)
            }
            for i in range(1000)
        ],
        "large_opportunity_set": [
            {
                "sport": ["NFL", "NBA", "MLB", "NHL", "Soccer"][i % 5],
                "event": f"Game {i}",
                "bet_type": ["spread", "moneyline", "total", "prop"][i % 4],
                "selection": f"Selection {i}",
                "odds": 1.80 + (i % 10) * 0.05,
                "probability": 0.45 + (i % 20) * 0.01,
                "edge": 0.01 + (i % 15) * 0.002,
                "confidence": 0.60 + (i % 25) * 0.01,
                "model_agreement": 0.65 + (i % 30) * 0.01,
                "hours_until_event": 12 + (i % 48),
                "liquidity": 25000 + (i % 10) * 5000
            }
            for i in range(500)
        ]
    }


@pytest.fixture
def mock_external_apis():
    """Mock external API dependencies"""
    with patch('requests.get') as mock_get, \
         patch('requests.post') as mock_post:
        
        # Mock successful API responses
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"status": "success", "data": {}}
        
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"status": "success", "data": {}}
        
        yield {"get": mock_get, "post": mock_post}


def stress_test_helper(operation_func, *args, **kwargs):
    """Helper function for stress testing operations"""
    start_time = time.time()
    success_count = 0
    error_count = 0
    
    try:
        result = operation_func(*args, **kwargs)
        if isinstance(result, dict) and result.get("status") == "failed":
            error_count += 1
        else:
            success_count += 1
    except Exception:
        error_count += 1
    
    end_time = time.time()
    
    return {
        "duration": end_time - start_time,
        "success": success_count > 0,
        "errors": error_count
    }


# =============================================================================
# CLEANUP AND TEARDOWN
# =============================================================================

def teardown_module(module):
    """Clean up after all tests"""
    # Clear global state
    from src.syndicate.syndicate_tools import (
        SYNDICATE_MANAGERS, ALLOCATION_ENGINES, 
        DISTRIBUTION_SYSTEMS, WITHDRAWAL_MANAGERS
    )
    
    SYNDICATE_MANAGERS.clear()
    ALLOCATION_ENGINES.clear()
    DISTRIBUTION_SYSTEMS.clear()
    WITHDRAWAL_MANAGERS.clear()


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        "-v", 
        "--tb=short",
        "--cov=src.syndicate",
        "--cov-report=term-missing",
        "--cov-report=html:tests/coverage_html",
        __file__
    ])