"""
Test fixtures for Fantasy Collective (Syndicate) System
Provides reusable test data and mock objects
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, MagicMock
from typing import Dict, List, Any
import json


@pytest.fixture
def mock_database():
    """Mock database for testing"""
    class MockDB:
        def __init__(self):
            self.syndicates = {}
            self.members = {}
            self.allocations = []
            self.distributions = []
            self.withdrawals = []
        
        def create_syndicate(self, syndicate_id: str, name: str, description: str = ""):
            if syndicate_id in self.syndicates:
                return {"error": f"Syndicate {syndicate_id} already exists"}
            
            self.syndicates[syndicate_id] = {
                "id": syndicate_id,
                "name": name,
                "description": description,
                "created_at": datetime.now(),
                "status": "active",
                "total_capital": Decimal("0")
            }
            return {"status": "success", "id": syndicate_id}
        
        def add_member(self, syndicate_id: str, member_data: Dict):
            if syndicate_id not in self.syndicates:
                return {"error": f"Syndicate {syndicate_id} not found"}
            
            member_id = f"member_{len(self.members) + 1}"
            self.members[member_id] = {
                "id": member_id,
                "syndicate_id": syndicate_id,
                **member_data,
                "joined_at": datetime.now(),
                "status": "active"
            }
            return {"status": "success", "member_id": member_id}
    
    return MockDB()


@pytest.fixture
def sample_syndicate_data():
    """Sample syndicate data for testing"""
    return {
        "syndicate_id": "test-syndicate-001",
        "name": "Elite Trading Collective",
        "description": "A high-performance trading syndicate focusing on sports betting and financial markets",
        "max_members": 25,
        "min_investment": 10000.0,
        "max_investment": 500000.0,
        "fee_structure": {
            "management_fee": 0.02,  # 2% annual
            "performance_fee": 0.20  # 20% of profits
        }
    }


@pytest.fixture
def sample_members_data():
    """Sample member data for testing"""
    return [
        {
            "name": "Alice Johnson",
            "email": "alice.johnson@example.com",
            "role": "lead_investor",
            "initial_contribution": 100000.0,
            "risk_tolerance": "moderate",
            "investment_experience": "expert"
        },
        {
            "name": "Bob Smith",
            "email": "bob.smith@example.com", 
            "role": "senior_analyst",
            "initial_contribution": 75000.0,
            "risk_tolerance": "aggressive",
            "investment_experience": "advanced"
        },
        {
            "name": "Charlie Brown",
            "email": "charlie.brown@example.com",
            "role": "junior_analyst", 
            "initial_contribution": 50000.0,
            "risk_tolerance": "conservative",
            "investment_experience": "intermediate"
        },
        {
            "name": "Diana Prince",
            "email": "diana.prince@example.com",
            "role": "contributing_member",
            "initial_contribution": 25000.0,
            "risk_tolerance": "moderate",
            "investment_experience": "beginner"
        },
        {
            "name": "Eve Adams",
            "email": "eve.adams@example.com",
            "role": "observer",
            "initial_contribution": 5000.0,
            "risk_tolerance": "conservative",
            "investment_experience": "beginner"
        }
    ]


@pytest.fixture
def sample_betting_opportunities():
    """Sample betting opportunities for testing"""
    return [
        {
            "id": "opp_001",
            "sport": "NFL",
            "league": "National Football League",
            "event": "Kansas City Chiefs vs Buffalo Bills",
            "event_date": datetime.now() + timedelta(days=2),
            "bet_type": "spread",
            "selection": "Kansas City Chiefs -3.0",
            "odds": 1.91,
            "probability": 0.58,
            "edge": 0.052,
            "confidence": 0.78,
            "model_agreement": 0.85,
            "liquidity": 150000,
            "market_efficiency": 0.92,
            "value_rating": "high",
            "risk_level": "medium"
        },
        {
            "id": "opp_002", 
            "sport": "NBA",
            "league": "National Basketball Association",
            "event": "Los Angeles Lakers vs Boston Celtics",
            "event_date": datetime.now() + timedelta(hours=18),
            "bet_type": "total",
            "selection": "Over 225.5 Points",
            "odds": 1.95,
            "probability": 0.55,
            "edge": 0.067,
            "confidence": 0.82,
            "model_agreement": 0.88,
            "liquidity": 200000,
            "market_efficiency": 0.89,
            "value_rating": "very_high",
            "risk_level": "low"
        },
        {
            "id": "opp_003",
            "sport": "MLB", 
            "league": "Major League Baseball",
            "event": "New York Yankees vs Boston Red Sox",
            "event_date": datetime.now() + timedelta(days=1),
            "bet_type": "moneyline",
            "selection": "New York Yankees -125",
            "odds": 1.80,
            "probability": 0.62,
            "edge": 0.033,
            "confidence": 0.71,
            "model_agreement": 0.76,
            "liquidity": 100000,
            "market_efficiency": 0.94,
            "value_rating": "medium",
            "risk_level": "medium"
        },
        {
            "id": "opp_004",
            "sport": "Soccer",
            "league": "Premier League",
            "event": "Manchester United vs Liverpool",
            "event_date": datetime.now() + timedelta(days=3),
            "bet_type": "both_teams_to_score",
            "selection": "Yes",
            "odds": 1.85,
            "probability": 0.67,
            "edge": 0.089,
            "confidence": 0.85,
            "model_agreement": 0.91,
            "liquidity": 175000,
            "market_efficiency": 0.87,
            "value_rating": "very_high",
            "risk_level": "low"
        },
        {
            "id": "opp_005",
            "sport": "Tennis",
            "league": "ATP Tour",
            "event": "Novak Djokovic vs Rafael Nadal",
            "event_date": datetime.now() + timedelta(hours=36),
            "bet_type": "match_winner",
            "selection": "Novak Djokovic",
            "odds": 2.10,
            "probability": 0.52,
            "edge": 0.021,
            "confidence": 0.63,
            "model_agreement": 0.68,
            "liquidity": 80000,
            "market_efficiency": 0.96,
            "value_rating": "low",
            "risk_level": "high"
        }
    ]


@pytest.fixture
def sample_allocation_strategies():
    """Sample allocation strategy configurations"""
    return {
        "kelly_criterion": {
            "name": "Kelly Criterion",
            "description": "Mathematical optimal betting based on edge and bankroll",
            "parameters": {
                "fractional_kelly": 0.25,  # Use 25% of full Kelly
                "max_bet_percentage": 0.05,  # Maximum 5% of bankroll per bet
                "confidence_multiplier": True,  # Adjust by confidence level
                "model_agreement_weight": 0.3  # Weight for model agreement
            },
            "risk_controls": {
                "max_daily_exposure": 0.20,
                "max_single_sport": 0.40,
                "min_edge_required": 0.02,
                "min_confidence_required": 0.60
            }
        },
        "fixed_percentage": {
            "name": "Fixed Percentage",
            "description": "Fixed percentage of bankroll per bet",
            "parameters": {
                "base_percentage": 0.02,  # 2% of bankroll per bet
                "scaling_factor": 1.5,  # Scale by edge/confidence
                "max_bet_percentage": 0.04  # Maximum 4% for high-confidence bets
            },
            "risk_controls": {
                "max_daily_exposure": 0.15,
                "max_single_sport": 0.35,
                "min_edge_required": 0.015,
                "min_confidence_required": 0.55
            }
        },
        "dynamic_confidence": {
            "name": "Dynamic Confidence",
            "description": "Bet size scales with confidence and edge",
            "parameters": {
                "base_percentage": 0.015,
                "confidence_multiplier": 2.0,  # Double bet size for high confidence
                "edge_multiplier": 1.5,  # Scale by edge magnitude
                "model_agreement_bonus": 0.3  # Bonus for high model agreement
            },
            "risk_controls": {
                "max_daily_exposure": 0.18,
                "max_single_sport": 0.38,
                "min_edge_required": 0.025,
                "min_confidence_required": 0.65
            }
        }
    }


@pytest.fixture
def sample_performance_metrics():
    """Sample performance metrics for testing"""
    return {
        "member_1": {
            "total_profit": 12500.50,
            "total_invested": 85000.00,
            "roi": 0.147,  # 14.7%
            "win_rate": 0.64,  # 64% win rate
            "average_return": 0.089,  # 8.9% average return per bet
            "sharpe_ratio": 1.23,
            "alpha": 0.056,  # 5.6% alpha
            "beta": 0.78,
            "max_drawdown": 0.085,  # 8.5% max drawdown
            "consecutive_wins": 7,
            "consecutive_losses": 3,
            "total_bets": 125,
            "winning_bets": 80,
            "losing_bets": 45,
            "skill_score": 0.78,
            "consistency_score": 0.82,
            "risk_score": 0.34
        },
        "member_2": {
            "total_profit": 8750.25,
            "total_invested": 65000.00,
            "roi": 0.135,  # 13.5%
            "win_rate": 0.58,
            "average_return": 0.092,
            "sharpe_ratio": 1.15,
            "alpha": 0.041,
            "beta": 0.85,
            "max_drawdown": 0.12,
            "consecutive_wins": 5,
            "consecutive_losses": 4,
            "total_bets": 98,
            "winning_bets": 57,
            "losing_bets": 41,
            "skill_score": 0.71,
            "consistency_score": 0.76,
            "risk_score": 0.42
        }
    }


@pytest.fixture
def sample_historical_data():
    """Sample historical data for backtesting and analysis"""
    base_date = datetime.now() - timedelta(days=90)
    
    return {
        "daily_performance": [
            {
                "date": base_date + timedelta(days=i),
                "daily_pnl": 150.25 * (1 + (i % 7 - 3) * 0.1),  # Vary by day
                "bets_placed": 2 + (i % 5),
                "win_rate": 0.60 + (i % 20) * 0.01,
                "total_staked": 2500.0 + (i % 10) * 100,
                "bankroll": 50000 + i * 25.5  # Growing bankroll
            }
            for i in range(90)
        ],
        "bet_history": [
            {
                "bet_id": f"bet_{i:04d}",
                "date": base_date + timedelta(days=i // 3),
                "sport": ["NFL", "NBA", "MLB", "NHL", "Soccer"][i % 5],
                "event": f"Game {i}",
                "selection": f"Selection {i}",
                "odds": 1.80 + (i % 15) * 0.05,
                "stake": 1000 + (i % 20) * 50,
                "result": "won" if i % 3 != 0 else "lost",  # ~67% win rate
                "profit_loss": (1000 + (i % 20) * 50) * (0.80 if i % 3 != 0 else -1),
                "confidence": 0.55 + (i % 30) * 0.01,
                "edge": 0.02 + (i % 25) * 0.002
            }
            for i in range(200)
        ]
    }


@pytest.fixture
def mock_external_data_feeds():
    """Mock external data feeds for testing"""
    return {
        "odds_feed": Mock(
            get_odds=Mock(return_value={
                "event_id": "test_event_001",
                "odds": {"home": 1.95, "away": 1.87, "draw": 3.40},
                "last_updated": datetime.now(),
                "bookmaker": "test_book"
            })
        ),
        "results_feed": Mock(
            get_result=Mock(return_value={
                "event_id": "test_event_001", 
                "result": "home",
                "score": "2-1",
                "status": "final"
            })
        ),
        "news_feed": Mock(
            get_news=Mock(return_value=[
                {
                    "title": "Team News Update",
                    "content": "Key player injury affects odds",
                    "sentiment": 0.75,
                    "relevance": 0.89,
                    "timestamp": datetime.now()
                }
            ])
        )
    }


@pytest.fixture
def sample_risk_scenarios():
    """Sample risk scenarios for testing"""
    return [
        {
            "name": "market_crash",
            "description": "Sudden market downturn affecting multiple sports",
            "probability": 0.05,
            "impact": {
                "bankroll_loss": 0.15,  # 15% bankroll loss
                "duration_days": 7,
                "affected_sports": ["NFL", "NBA", "MLB"],
                "recovery_time": 14
            }
        },
        {
            "name": "model_failure", 
            "description": "Prediction model becomes unreliable",
            "probability": 0.08,
            "impact": {
                "accuracy_drop": 0.20,  # 20% drop in accuracy
                "duration_days": 3,
                "affected_strategies": ["kelly_criterion", "dynamic_confidence"],
                "recovery_time": 5
            }
        },
        {
            "name": "liquidity_crisis",
            "description": "Reduced market liquidity and higher spreads",
            "probability": 0.12,
            "impact": {
                "spread_increase": 0.30,  # 30% wider spreads
                "available_markets": 0.60,  # 40% fewer markets
                "duration_days": 2,
                "recovery_time": 1
            }
        },
        {
            "name": "regulatory_change",
            "description": "New regulations affecting betting operations",
            "probability": 0.03,
            "impact": {
                "operational_cost": 0.05,  # 5% increase in costs
                "market_access": 0.80,  # 20% reduction in available markets
                "compliance_time": 30,
                "adaptation_cost": 10000
            }
        }
    ]


@pytest.fixture 
def load_test_configuration():
    """Configuration for load testing"""
    return {
        "concurrent_users": [10, 50, 100, 200],
        "test_duration_seconds": [30, 60, 120],
        "operations": [
            "create_syndicate",
            "add_member", 
            "get_status",
            "allocate_funds",
            "distribute_profits",
            "process_withdrawal"
        ],
        "success_rate_threshold": 0.95,  # 95% success rate required
        "response_time_threshold": 2.0,  # 2 seconds max response time
        "memory_threshold": 500 * 1024 * 1024,  # 500MB max memory usage
        "cpu_threshold": 80.0  # 80% max CPU usage
    }


@pytest.fixture
def database_test_scenarios():
    """Database test scenarios for validation"""
    return {
        "transaction_scenarios": [
            {
                "name": "successful_multi_operation",
                "operations": [
                    {"type": "create_syndicate", "data": {"id": "tx_001", "name": "TX Test"}},
                    {"type": "add_member", "data": {"syndicate_id": "tx_001", "name": "User1"}},
                    {"type": "add_member", "data": {"syndicate_id": "tx_001", "name": "User2"}}
                ],
                "expected_result": "success"
            },
            {
                "name": "failed_operation_rollback",
                "operations": [
                    {"type": "create_syndicate", "data": {"id": "tx_002", "name": "TX Test 2"}},
                    {"type": "add_member", "data": {"syndicate_id": "tx_002", "name": "User1"}},
                    {"type": "add_member", "data": {"syndicate_id": "INVALID", "name": "User2"}}  # This should fail
                ],
                "expected_result": "rollback"
            }
        ],
        "constraint_tests": [
            {
                "name": "unique_email_constraint",
                "operations": [
                    {"type": "add_member", "data": {"email": "unique@test.com", "name": "User1"}},
                    {"type": "add_member", "data": {"email": "unique@test.com", "name": "User2"}}  # Should fail
                ],
                "expected_constraint_violation": "unique_email"
            },
            {
                "name": "foreign_key_constraint",
                "operations": [
                    {"type": "add_member", "data": {"syndicate_id": "nonexistent", "name": "User1"}}
                ],
                "expected_constraint_violation": "foreign_key"
            }
        ]
    }


@pytest.fixture
def security_test_vectors():
    """Security test vectors for comprehensive security testing"""
    return {
        "sql_injection_vectors": [
            "'; DROP TABLE syndicates; --",
            "' OR '1'='1'; --",
            "admin'; DELETE FROM members WHERE '1'='1'; --",
            "' UNION SELECT password FROM users WHERE username='admin'; --",
            "'; INSERT INTO members (role) VALUES ('admin'); --"
        ],
        "xss_vectors": [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "'-alert('XSS')-'"
        ],
        "path_traversal_vectors": [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ],
        "oversized_input_vectors": [
            "A" * 10000,  # 10KB string
            "A" * 100000,  # 100KB string
            "A" * 1000000,  # 1MB string
        ],
        "special_character_vectors": [
            "test\x00null",
            "test\r\ninjection",
            "test\tcontrol",
            "test\x1funit_separator",
            "test\x7fdelete"
        ]
    }


@pytest.fixture(autouse=True)
def cleanup_global_state():
    """Automatically clean up global state after each test"""
    yield
    
    # Clean up any global state that might persist between tests
    try:
        from src.syndicate.syndicate_tools import (
            SYNDICATE_MANAGERS, ALLOCATION_ENGINES,
            DISTRIBUTION_SYSTEMS, WITHDRAWAL_MANAGERS
        )
        
        # Clear all global managers
        for manager_dict in [SYNDICATE_MANAGERS, ALLOCATION_ENGINES,
                           DISTRIBUTION_SYSTEMS, WITHDRAWAL_MANAGERS]:
            if hasattr(manager_dict, 'clear'):
                manager_dict.clear()
    except ImportError:
        # Module might not be available in all test contexts
        pass


# Utility functions for test data generation

def generate_realistic_member_data(count: int = 10) -> List[Dict]:
    """Generate realistic member data for testing"""
    import random
    
    first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Iris", "Jack"]
    last_names = ["Smith", "Johnson", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor", "Anderson", "Thomas"]
    roles = ["lead_investor", "senior_analyst", "junior_analyst", "contributing_member", "observer"]
    domains = ["gmail.com", "yahoo.com", "hotmail.com", "company.com", "trading.com"]
    
    members = []
    for i in range(count):
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        role = random.choice(roles)
        domain = random.choice(domains)
        
        # Role-based contribution ranges
        contribution_ranges = {
            "lead_investor": (75000, 500000),
            "senior_analyst": (40000, 150000), 
            "junior_analyst": (20000, 75000),
            "contributing_member": (10000, 50000),
            "observer": (1000, 10000)
        }
        
        min_contrib, max_contrib = contribution_ranges[role]
        contribution = random.uniform(min_contrib, max_contrib)
        
        members.append({
            "name": f"{first_name} {last_name}",
            "email": f"{first_name.lower()}.{last_name.lower()}{i}@{domain}",
            "role": role,
            "initial_contribution": round(contribution, 2)
        })
    
    return members


def generate_market_opportunities(count: int = 20) -> List[Dict]:
    """Generate realistic market opportunities for testing"""
    import random
    
    sports = ["NFL", "NBA", "MLB", "NHL", "Soccer", "Tennis", "Golf", "Boxing", "MMA", "Cricket"]
    bet_types = ["spread", "moneyline", "total", "prop", "futures", "live"]
    
    opportunities = []
    base_date = datetime.now()
    
    for i in range(count):
        sport = random.choice(sports)
        bet_type = random.choice(bet_types)
        
        # Generate realistic odds and probabilities
        true_prob = random.uniform(0.35, 0.75)
        market_prob = true_prob + random.uniform(-0.05, 0.05)  # Market inefficiency
        odds = 1 / max(market_prob, 0.05)  # Prevent division by zero
        
        edge = max(0, true_prob - market_prob)
        confidence = random.uniform(0.5, 0.95)
        model_agreement = random.uniform(0.6, 0.95)
        
        opportunities.append({
            "id": f"opp_{i:04d}",
            "sport": sport,
            "event": f"{sport} Event {i}",
            "bet_type": bet_type,
            "selection": f"Selection {i}",
            "odds": round(odds, 2),
            "probability": round(true_prob, 3),
            "edge": round(edge, 4),
            "confidence": round(confidence, 3),
            "model_agreement": round(model_agreement, 3),
            "event_date": base_date + timedelta(hours=random.randint(1, 168)),  # 1 hour to 1 week
            "liquidity": random.randint(10000, 200000),
            "market_efficiency": random.uniform(0.85, 0.98)
        })
    
    return opportunities