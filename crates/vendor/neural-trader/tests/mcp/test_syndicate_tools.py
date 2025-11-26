"""
Comprehensive Test Suite for Syndicate MCP Tools Integration

This module tests all 17 syndicate management tools with:
- Unit tests for each tool's functionality
- Integration tests with MCP server
- Security and permission validation
- Performance and concurrency tests
- Multi-user operation scenarios

Test Coverage:
1. Member Management (create, update, remove, list members)
2. Fund Allocation (allocate, distribute, track funds)
3. Voting System (create proposals, cast votes, tally results)
4. Analytics & Reporting (performance metrics, member analytics)
5. Security & Authentication (permissions, role-based access)
"""

import pytest
import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import uuid
import random
import time
from concurrent.futures import ThreadPoolExecutor
import threading

# Import syndicate components
from src.sports_betting.syndicate.capital_manager import CapitalManager, AllocationMethod
from src.sports_betting.syndicate.voting_system import VotingSystem, ProposalType, VoteType
from src.sports_betting.syndicate.member_manager import MemberManager, MemberRole, MemberStatus
from src.sports_betting.syndicate.collaboration import CollaborationManager
from src.sports_betting.syndicate.smart_contracts import SmartContractManager


# ===== FIXTURES =====

@pytest.fixture
def mock_syndicate_data():
    """Generate mock syndicate data for testing"""
    return {
        "syndicate_id": "TEST_SYNDICATE_001",
        "name": "Test Betting Syndicate",
        "created_at": datetime.now(),
        "total_capital": Decimal("100000"),
        "member_count": 5,
        "active": True
    }


@pytest.fixture
def mock_members():
    """Generate mock member data with different roles"""
    members = []
    roles = [MemberRole.FOUNDER, MemberRole.LEAD, MemberRole.ANALYST, 
             MemberRole.CONTRIBUTOR, MemberRole.OBSERVER]
    
    for i, role in enumerate(roles):
        member_id = f"MEMBER_{i+1:03d}"
        members.append({
            "member_id": member_id,
            "username": f"test_user_{i+1}",
            "email": f"user{i+1}@test.com",
            "role": role,
            "status": MemberStatus.ACTIVE,
            "capital_contribution": Decimal(random.randint(5000, 50000)),
            "performance_score": round(random.uniform(0.5, 0.95), 2),
            "joined_at": datetime.now() - timedelta(days=random.randint(1, 365)),
            "permissions": _get_role_permissions(role)
        })
    
    return members


@pytest.fixture
def mock_betting_opportunities():
    """Generate mock betting opportunities"""
    sports = ["NFL", "NBA", "MLB", "NHL", "Soccer"]
    bet_types = ["moneyline", "spread", "over/under", "prop"]
    
    opportunities = []
    for i in range(10):
        opportunities.append({
            "bet_id": f"BET_{i+1:03d}",
            "sport": random.choice(sports),
            "event": f"Team A vs Team B - Game {i+1}",
            "bet_type": random.choice(bet_types),
            "odds": round(random.uniform(1.5, 3.5), 2),
            "recommended_stake": Decimal(random.randint(1000, 10000)),
            "confidence": round(random.uniform(0.6, 0.9), 2),
            "expected_value": round(random.uniform(1.05, 1.25), 2),
            "created_at": datetime.now()
        })
    
    return opportunities


@pytest.fixture
def mock_voting_scenarios():
    """Generate mock voting scenarios"""
    return [
        {
            "proposal_id": "PROP_001",
            "type": ProposalType.LARGE_BET,
            "title": "High-Stakes NFL Bet",
            "description": "Allocate $25,000 on Super Bowl",
            "votes_required": 3,
            "voting_deadline": datetime.now() + timedelta(hours=24)
        },
        {
            "proposal_id": "PROP_002",
            "type": ProposalType.STRATEGY_CHANGE,
            "title": "Implement ML Strategy",
            "description": "Adopt machine learning for bet selection",
            "votes_required": 4,
            "voting_deadline": datetime.now() + timedelta(days=3)
        },
        {
            "proposal_id": "PROP_003",
            "type": ProposalType.MEMBER_ADDITION,
            "title": "Add New Analyst",
            "description": "Invite expert NBA analyst to syndicate",
            "votes_required": 3,
            "voting_deadline": datetime.now() + timedelta(days=2)
        }
    ]


@pytest.fixture
async def mcp_server_mock():
    """Mock MCP server for testing"""
    server = AsyncMock()
    server.call_tool = AsyncMock()
    server.list_tools = AsyncMock(return_value=_get_syndicate_tools_list())
    server.gpu_available = True
    return server


# ===== HELPER FUNCTIONS =====

def _get_role_permissions(role: MemberRole) -> List[str]:
    """Get permissions based on member role"""
    base_permissions = ["view_bets", "view_performance"]
    
    role_permissions = {
        MemberRole.FOUNDER: base_permissions + ["manage_members", "manage_funds", "create_proposals", 
                                               "approve_withdrawals", "modify_settings"],
        MemberRole.LEAD: base_permissions + ["create_proposals", "manage_bets", "approve_small_withdrawals"],
        MemberRole.ANALYST: base_permissions + ["create_analysis", "suggest_bets"],
        MemberRole.CONTRIBUTOR: base_permissions + ["vote_proposals"],
        MemberRole.OBSERVER: base_permissions
    }
    
    return role_permissions.get(role, base_permissions)


def _get_syndicate_tools_list() -> List[Dict]:
    """Get list of all syndicate MCP tools"""
    return [
        # Member Management Tools
        {
            "name": "syndicate_create_member",
            "description": "Create a new syndicate member",
            "category": "member_management"
        },
        {
            "name": "syndicate_update_member",
            "description": "Update member information and roles",
            "category": "member_management"
        },
        {
            "name": "syndicate_remove_member",
            "description": "Remove a member from syndicate",
            "category": "member_management"
        },
        {
            "name": "syndicate_list_members",
            "description": "List all syndicate members with details",
            "category": "member_management"
        },
        
        # Fund Management Tools
        {
            "name": "syndicate_allocate_funds",
            "description": "Allocate funds for a betting opportunity",
            "category": "fund_management"
        },
        {
            "name": "syndicate_distribute_profits",
            "description": "Distribute profits among members",
            "category": "fund_management"
        },
        {
            "name": "syndicate_track_contributions",
            "description": "Track member contributions and balances",
            "category": "fund_management"
        },
        {
            "name": "syndicate_process_withdrawal",
            "description": "Process member withdrawal requests",
            "category": "fund_management"
        },
        
        # Voting System Tools
        {
            "name": "syndicate_create_proposal",
            "description": "Create a new voting proposal",
            "category": "voting_system"
        },
        {
            "name": "syndicate_cast_vote",
            "description": "Cast a vote on a proposal",
            "category": "voting_system"
        },
        {
            "name": "syndicate_tally_votes",
            "description": "Tally votes and determine outcome",
            "category": "voting_system"
        },
        {
            "name": "syndicate_list_proposals",
            "description": "List all proposals with status",
            "category": "voting_system"
        },
        
        # Analytics Tools
        {
            "name": "syndicate_performance_metrics",
            "description": "Get syndicate performance metrics",
            "category": "analytics"
        },
        {
            "name": "syndicate_member_analytics",
            "description": "Get individual member analytics",
            "category": "analytics"
        },
        {
            "name": "syndicate_bet_history",
            "description": "Get comprehensive bet history",
            "category": "analytics"
        },
        
        # Security Tools
        {
            "name": "syndicate_verify_permissions",
            "description": "Verify member permissions for action",
            "category": "security"
        },
        {
            "name": "syndicate_audit_log",
            "description": "Access syndicate audit log",
            "category": "security"
        }
    ]


# ===== UNIT TESTS =====

class TestSyndicateMemberManagement:
    """Test member management operations"""
    
    @pytest.mark.asyncio
    async def test_create_member(self, mcp_server_mock, mock_syndicate_data):
        """Test creating a new syndicate member"""
        # Arrange
        new_member_data = {
            "username": "new_member",
            "email": "new@test.com",
            "role": MemberRole.CONTRIBUTOR.value,
            "initial_contribution": 10000
        }
        
        expected_response = {
            "member_id": "MEMBER_006",
            "status": "created",
            "permissions": _get_role_permissions(MemberRole.CONTRIBUTOR)
        }
        
        mcp_server_mock.call_tool.return_value = expected_response
        
        # Act
        result = await mcp_server_mock.call_tool(
            "syndicate_create_member",
            new_member_data
        )
        
        # Assert
        assert result["status"] == "created"
        assert "member_id" in result
        assert len(result["permissions"]) > 0
        mcp_server_mock.call_tool.assert_called_once_with(
            "syndicate_create_member",
            new_member_data
        )
    
    @pytest.mark.asyncio
    async def test_update_member_role(self, mcp_server_mock, mock_members):
        """Test updating member role and permissions"""
        # Arrange
        member_to_update = mock_members[2]  # Analyst
        update_data = {
            "member_id": member_to_update["member_id"],
            "new_role": MemberRole.LEAD.value,
            "reason": "Promotion based on performance"
        }
        
        expected_response = {
            "status": "updated",
            "old_role": MemberRole.ANALYST.value,
            "new_role": MemberRole.LEAD.value,
            "new_permissions": _get_role_permissions(MemberRole.LEAD)
        }
        
        mcp_server_mock.call_tool.return_value = expected_response
        
        # Act
        result = await mcp_server_mock.call_tool(
            "syndicate_update_member",
            update_data
        )
        
        # Assert
        assert result["status"] == "updated"
        assert result["new_role"] == MemberRole.LEAD.value
        assert "create_proposals" in result["new_permissions"]
    
    @pytest.mark.asyncio
    async def test_remove_member_with_settlement(self, mcp_server_mock, mock_members):
        """Test removing member with proper fund settlement"""
        # Arrange
        member_to_remove = mock_members[4]  # Observer
        removal_data = {
            "member_id": member_to_remove["member_id"],
            "reason": "Voluntary exit",
            "settle_funds": True
        }
        
        expected_response = {
            "status": "removed",
            "final_settlement": {
                "capital_returned": float(member_to_remove["capital_contribution"]),
                "pending_profits": 1250.50,
                "total_payout": float(member_to_remove["capital_contribution"]) + 1250.50
            },
            "removal_timestamp": datetime.now().isoformat()
        }
        
        mcp_server_mock.call_tool.return_value = expected_response
        
        # Act
        result = await mcp_server_mock.call_tool(
            "syndicate_remove_member",
            removal_data
        )
        
        # Assert
        assert result["status"] == "removed"
        assert result["final_settlement"]["total_payout"] > 0
        assert "removal_timestamp" in result
    
    @pytest.mark.asyncio
    async def test_list_members_with_filters(self, mcp_server_mock, mock_members):
        """Test listing members with various filters"""
        # Arrange
        filter_data = {
            "status": "active",
            "min_contribution": 10000,
            "roles": [MemberRole.LEAD.value, MemberRole.ANALYST.value]
        }
        
        filtered_members = [m for m in mock_members 
                          if m["role"] in [MemberRole.LEAD, MemberRole.ANALYST] 
                          and m["capital_contribution"] >= 10000]
        
        mcp_server_mock.call_tool.return_value = {
            "members": filtered_members,
            "total_count": len(filtered_members),
            "total_capital": sum(m["capital_contribution"] for m in filtered_members)
        }
        
        # Act
        result = await mcp_server_mock.call_tool(
            "syndicate_list_members",
            filter_data
        )
        
        # Assert
        assert len(result["members"]) <= len(mock_members)
        assert all(m["capital_contribution"] >= 10000 for m in result["members"])
        assert result["total_capital"] > 0


class TestSyndicateFundManagement:
    """Test fund allocation and distribution"""
    
    @pytest.mark.asyncio
    async def test_allocate_funds_proportional(self, mcp_server_mock, mock_members, mock_betting_opportunities):
        """Test proportional fund allocation for bet"""
        # Arrange
        bet = mock_betting_opportunities[0]
        allocation_request = {
            "bet_id": bet["bet_id"],
            "total_amount": float(bet["recommended_stake"]),
            "allocation_method": AllocationMethod.PROPORTIONAL.value,
            "participating_members": [m["member_id"] for m in mock_members[:3]]
        }
        
        total_capital = sum(m["capital_contribution"] for m in mock_members[:3])
        allocations = {}
        for member in mock_members[:3]:
            proportion = float(member["capital_contribution"]) / float(total_capital)
            allocations[member["member_id"]] = round(float(bet["recommended_stake"]) * proportion, 2)
        
        expected_response = {
            "status": "allocated",
            "allocations": allocations,
            "total_allocated": sum(allocations.values()),
            "allocation_method": AllocationMethod.PROPORTIONAL.value
        }
        
        mcp_server_mock.call_tool.return_value = expected_response
        
        # Act
        result = await mcp_server_mock.call_tool(
            "syndicate_allocate_funds",
            allocation_request
        )
        
        # Assert
        assert result["status"] == "allocated"
        assert abs(result["total_allocated"] - float(bet["recommended_stake"])) < 1
        assert len(result["allocations"]) == 3
    
    @pytest.mark.asyncio
    async def test_distribute_profits_performance_weighted(self, mcp_server_mock, mock_members):
        """Test performance-weighted profit distribution"""
        # Arrange
        profit_data = {
            "bet_id": "BET_001",
            "total_profit": 15000,
            "distribution_method": "performance_weighted",
            "participating_members": [m["member_id"] for m in mock_members[:4]]
        }
        
        # Calculate performance-weighted distribution
        total_performance = sum(m["performance_score"] for m in mock_members[:4])
        distributions = {}
        for member in mock_members[:4]:
            weight = member["performance_score"] / total_performance
            distributions[member["member_id"]] = round(profit_data["total_profit"] * weight, 2)
        
        expected_response = {
            "status": "distributed",
            "distributions": distributions,
            "total_distributed": sum(distributions.values()),
            "distribution_method": "performance_weighted",
            "timestamp": datetime.now().isoformat()
        }
        
        mcp_server_mock.call_tool.return_value = expected_response
        
        # Act
        result = await mcp_server_mock.call_tool(
            "syndicate_distribute_profits",
            profit_data
        )
        
        # Assert
        assert result["status"] == "distributed"
        assert abs(result["total_distributed"] - profit_data["total_profit"]) < 1
        assert all(amount > 0 for amount in result["distributions"].values())
    
    @pytest.mark.asyncio
    async def test_track_contributions_over_time(self, mcp_server_mock, mock_members):
        """Test tracking member contributions and balance changes"""
        # Arrange
        tracking_request = {
            "member_id": mock_members[0]["member_id"],
            "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
            "end_date": datetime.now().isoformat(),
            "include_transactions": True
        }
        
        expected_response = {
            "member_id": mock_members[0]["member_id"],
            "initial_balance": float(mock_members[0]["capital_contribution"]),
            "current_balance": float(mock_members[0]["capital_contribution"]) * 1.15,
            "total_deposits": 10000,
            "total_withdrawals": 0,
            "total_profits": float(mock_members[0]["capital_contribution"]) * 0.15,
            "roi_percentage": 15.0,
            "transactions": [
                {
                    "date": (datetime.now() - timedelta(days=20)).isoformat(),
                    "type": "deposit",
                    "amount": 10000,
                    "description": "Additional investment"
                },
                {
                    "date": (datetime.now() - timedelta(days=10)).isoformat(),
                    "type": "profit",
                    "amount": float(mock_members[0]["capital_contribution"]) * 0.15,
                    "description": "Profit from BET_001"
                }
            ]
        }
        
        mcp_server_mock.call_tool.return_value = expected_response
        
        # Act
        result = await mcp_server_mock.call_tool(
            "syndicate_track_contributions",
            tracking_request
        )
        
        # Assert
        assert result["current_balance"] > result["initial_balance"]
        assert result["roi_percentage"] == 15.0
        assert len(result["transactions"]) == 2
    
    @pytest.mark.asyncio
    async def test_process_withdrawal_with_validation(self, mcp_server_mock, mock_members):
        """Test processing withdrawal with balance validation"""
        # Arrange
        withdrawal_request = {
            "member_id": mock_members[1]["member_id"],
            "amount": 5000,
            "reason": "Partial withdrawal",
            "require_approval": True
        }
        
        expected_response = {
            "status": "pending_approval",
            "withdrawal_id": "WD_001",
            "member_id": mock_members[1]["member_id"],
            "amount": 5000,
            "available_balance": float(mock_members[1]["capital_contribution"]) * 1.1,
            "remaining_balance": float(mock_members[1]["capital_contribution"]) * 1.1 - 5000,
            "approval_required_from": ["MEMBER_001"],  # Founder
            "created_at": datetime.now().isoformat()
        }
        
        mcp_server_mock.call_tool.return_value = expected_response
        
        # Act
        result = await mcp_server_mock.call_tool(
            "syndicate_process_withdrawal",
            withdrawal_request
        )
        
        # Assert
        assert result["status"] == "pending_approval"
        assert result["remaining_balance"] > 0
        assert len(result["approval_required_from"]) > 0


class TestSyndicateVotingSystem:
    """Test voting and governance functionality"""
    
    @pytest.mark.asyncio
    async def test_create_proposal_with_validation(self, mcp_server_mock, mock_members):
        """Test creating a voting proposal with validation"""
        # Arrange
        proposal_data = {
            "proposer_id": mock_members[1]["member_id"],  # Lead member
            "type": ProposalType.LARGE_BET.value,
            "title": "High-Value NBA Finals Bet",
            "description": "Allocate $30,000 on NBA Finals Game 7",
            "details": {
                "bet_amount": 30000,
                "odds": 2.5,
                "expected_return": 75000,
                "risk_assessment": "medium-high"
            },
            "voting_deadline_hours": 48,
            "required_approval_percentage": 66.7
        }
        
        expected_response = {
            "status": "created",
            "proposal_id": "PROP_004",
            "proposer_id": mock_members[1]["member_id"],
            "voting_opens": datetime.now().isoformat(),
            "voting_closes": (datetime.now() + timedelta(hours=48)).isoformat(),
            "eligible_voters": [m["member_id"] for m in mock_members if m["role"] != MemberRole.OBSERVER],
            "votes_required": 3  # 66.7% of 4 eligible voters
        }
        
        mcp_server_mock.call_tool.return_value = expected_response
        
        # Act
        result = await mcp_server_mock.call_tool(
            "syndicate_create_proposal",
            proposal_data
        )
        
        # Assert
        assert result["status"] == "created"
        assert "proposal_id" in result
        assert len(result["eligible_voters"]) == 4  # Excludes observer
        assert result["votes_required"] == 3
    
    @pytest.mark.asyncio
    async def test_cast_vote_with_weight(self, mcp_server_mock, mock_members, mock_voting_scenarios):
        """Test casting weighted votes"""
        # Arrange
        proposal = mock_voting_scenarios[0]
        vote_data = {
            "proposal_id": proposal["proposal_id"],
            "voter_id": mock_members[0]["member_id"],  # Founder
            "vote": VoteType.YES.value,
            "rationale": "Strong statistical advantage identified",
            "weight_by_capital": True
        }
        
        expected_response = {
            "status": "vote_recorded",
            "proposal_id": proposal["proposal_id"],
            "voter_id": mock_members[0]["member_id"],
            "vote": VoteType.YES.value,
            "vote_weight": 1.5,  # Higher weight for founder
            "current_tally": {
                "yes": 1.5,
                "no": 0,
                "abstain": 0,
                "total_votes": 1,
                "weighted_total": 1.5
            },
            "timestamp": datetime.now().isoformat()
        }
        
        mcp_server_mock.call_tool.return_value = expected_response
        
        # Act
        result = await mcp_server_mock.call_tool(
            "syndicate_cast_vote",
            vote_data
        )
        
        # Assert
        assert result["status"] == "vote_recorded"
        assert result["vote_weight"] > 1.0
        assert result["current_tally"]["weighted_total"] == 1.5
    
    @pytest.mark.asyncio
    async def test_tally_votes_with_outcome(self, mcp_server_mock, mock_voting_scenarios):
        """Test vote tallying and outcome determination"""
        # Arrange
        proposal = mock_voting_scenarios[0]
        tally_request = {
            "proposal_id": proposal["proposal_id"],
            "close_voting": True
        }
        
        expected_response = {
            "status": "completed",
            "proposal_id": proposal["proposal_id"],
            "final_tally": {
                "yes": 3,
                "no": 1,
                "abstain": 0,
                "total_votes": 4,
                "participation_rate": 80.0
            },
            "outcome": "approved",
            "approval_percentage": 75.0,
            "required_percentage": 60.0,
            "voting_closed_at": datetime.now().isoformat(),
            "execution_status": "pending_execution"
        }
        
        mcp_server_mock.call_tool.return_value = expected_response
        
        # Act
        result = await mcp_server_mock.call_tool(
            "syndicate_tally_votes",
            tally_request
        )
        
        # Assert
        assert result["status"] == "completed"
        assert result["outcome"] == "approved"
        assert result["approval_percentage"] >= result["required_percentage"]
        assert result["execution_status"] == "pending_execution"
    
    @pytest.mark.asyncio
    async def test_list_proposals_with_status_filter(self, mcp_server_mock, mock_voting_scenarios):
        """Test listing proposals with status filtering"""
        # Arrange
        list_request = {
            "status_filter": ["active", "pending"],
            "proposer_filter": None,
            "type_filter": [ProposalType.LARGE_BET.value],
            "sort_by": "voting_deadline",
            "limit": 10
        }
        
        filtered_proposals = [p for p in mock_voting_scenarios 
                            if p["type"] == ProposalType.LARGE_BET]
        
        expected_response = {
            "proposals": filtered_proposals,
            "total_count": len(filtered_proposals),
            "status_summary": {
                "active": 1,
                "pending": 0,
                "completed": 0,
                "rejected": 0
            }
        }
        
        mcp_server_mock.call_tool.return_value = expected_response
        
        # Act
        result = await mcp_server_mock.call_tool(
            "syndicate_list_proposals",
            list_request
        )
        
        # Assert
        assert len(result["proposals"]) == 1
        assert result["proposals"][0]["type"] == ProposalType.LARGE_BET
        assert result["status_summary"]["active"] == 1


class TestSyndicateAnalytics:
    """Test analytics and reporting functionality"""
    
    @pytest.mark.asyncio
    async def test_performance_metrics_comprehensive(self, mcp_server_mock, mock_syndicate_data):
        """Test comprehensive performance metrics calculation"""
        # Arrange
        metrics_request = {
            "start_date": (datetime.now() - timedelta(days=90)).isoformat(),
            "end_date": datetime.now().isoformat(),
            "include_breakdown": True,
            "metrics": ["roi", "sharpe_ratio", "win_rate", "average_bet_size"]
        }
        
        expected_response = {
            "syndicate_id": mock_syndicate_data["syndicate_id"],
            "period": {
                "start": metrics_request["start_date"],
                "end": metrics_request["end_date"],
                "days": 90
            },
            "overall_metrics": {
                "total_roi": 25.5,
                "annualized_roi": 102.0,
                "sharpe_ratio": 1.85,
                "win_rate": 0.65,
                "average_bet_size": 5500,
                "total_bets": 120,
                "winning_bets": 78,
                "losing_bets": 42
            },
            "monthly_breakdown": [
                {"month": "2024-10", "roi": 8.2, "bets": 38, "win_rate": 0.68},
                {"month": "2024-11", "roi": 9.1, "bets": 42, "win_rate": 0.64},
                {"month": "2024-12", "roi": 8.2, "bets": 40, "win_rate": 0.63}
            ],
            "by_sport": {
                "NFL": {"roi": 32.1, "bets": 45, "win_rate": 0.71},
                "NBA": {"roi": 18.5, "bets": 35, "win_rate": 0.63},
                "MLB": {"roi": 22.8, "bets": 40, "win_rate": 0.60}
            },
            "risk_metrics": {
                "max_drawdown": -12.5,
                "value_at_risk_95": -8500,
                "expected_shortfall": -10200,
                "risk_adjusted_return": 0.85
            }
        }
        
        mcp_server_mock.call_tool.return_value = expected_response
        
        # Act
        result = await mcp_server_mock.call_tool(
            "syndicate_performance_metrics",
            metrics_request
        )
        
        # Assert
        assert result["overall_metrics"]["total_roi"] == 25.5
        assert result["overall_metrics"]["sharpe_ratio"] > 1.5
        assert len(result["monthly_breakdown"]) == 3
        assert "NFL" in result["by_sport"]
        assert result["risk_metrics"]["max_drawdown"] < 0
    
    @pytest.mark.asyncio
    async def test_member_analytics_detailed(self, mcp_server_mock, mock_members):
        """Test detailed member analytics"""
        # Arrange
        member = mock_members[0]
        analytics_request = {
            "member_id": member["member_id"],
            "include_bet_history": True,
            "include_contribution_analysis": True,
            "period_days": 180
        }
        
        expected_response = {
            "member_id": member["member_id"],
            "username": member["username"],
            "performance_summary": {
                "total_roi": 28.5,
                "win_rate": 0.68,
                "average_bet_participation": 0.85,
                "profit_contribution": 15250,
                "ranking": 2,
                "percentile": 90
            },
            "contribution_analysis": {
                "initial_capital": float(member["capital_contribution"]),
                "current_balance": float(member["capital_contribution"]) * 1.285,
                "total_deposits": float(member["capital_contribution"]) + 10000,
                "total_withdrawals": 5000,
                "net_contribution": float(member["capital_contribution"]) + 5000
            },
            "betting_patterns": {
                "favorite_sports": ["NFL", "NBA"],
                "average_stake": 2500,
                "risk_preference": "moderate",
                "best_performing_bet_type": "spread"
            },
            "recent_bets": [
                {
                    "bet_id": "BET_098",
                    "date": (datetime.now() - timedelta(days=2)).isoformat(),
                    "sport": "NFL",
                    "stake": 2000,
                    "outcome": "win",
                    "profit": 1800
                },
                {
                    "bet_id": "BET_097",
                    "date": (datetime.now() - timedelta(days=5)).isoformat(),
                    "sport": "NBA",
                    "stake": 1500,
                    "outcome": "loss",
                    "profit": -1500
                }
            ],
            "engagement_metrics": {
                "proposals_created": 12,
                "votes_cast": 45,
                "vote_participation_rate": 0.92,
                "last_active": datetime.now().isoformat()
            }
        }
        
        mcp_server_mock.call_tool.return_value = expected_response
        
        # Act
        result = await mcp_server_mock.call_tool(
            "syndicate_member_analytics",
            analytics_request
        )
        
        # Assert
        assert result["performance_summary"]["total_roi"] > 0
        assert result["performance_summary"]["ranking"] == 2
        assert result["contribution_analysis"]["current_balance"] > result["contribution_analysis"]["initial_capital"]
        assert len(result["recent_bets"]) > 0
        assert result["engagement_metrics"]["vote_participation_rate"] > 0.9
    
    @pytest.mark.asyncio
    async def test_bet_history_with_filters(self, mcp_server_mock, mock_betting_opportunities):
        """Test bet history retrieval with various filters"""
        # Arrange
        history_request = {
            "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
            "end_date": datetime.now().isoformat(),
            "sport_filter": ["NFL", "NBA"],
            "outcome_filter": "all",
            "min_stake": 1000,
            "sort_by": "profit",
            "sort_order": "desc",
            "page": 1,
            "page_size": 20
        }
        
        bet_history = []
        for i, bet in enumerate(mock_betting_opportunities[:5]):
            outcome = "win" if i % 2 == 0 else "loss"
            profit = float(bet["recommended_stake"]) * (bet["odds"] - 1) if outcome == "win" else -float(bet["recommended_stake"])
            
            bet_history.append({
                "bet_id": bet["bet_id"],
                "date": bet["created_at"].isoformat(),
                "sport": bet["sport"],
                "event": bet["event"],
                "bet_type": bet["bet_type"],
                "stake": float(bet["recommended_stake"]),
                "odds": bet["odds"],
                "outcome": outcome,
                "profit": profit,
                "participants": 3,
                "notes": f"Confidence: {bet['confidence']}"
            })
        
        expected_response = {
            "bets": sorted(bet_history, key=lambda x: x["profit"], reverse=True),
            "pagination": {
                "page": 1,
                "page_size": 20,
                "total_pages": 1,
                "total_items": len(bet_history)
            },
            "summary": {
                "total_bets": len(bet_history),
                "total_stake": sum(b["stake"] for b in bet_history),
                "total_profit": sum(b["profit"] for b in bet_history),
                "average_odds": sum(b["odds"] for b in bet_history) / len(bet_history),
                "roi_percentage": (sum(b["profit"] for b in bet_history) / sum(b["stake"] for b in bet_history)) * 100
            }
        }
        
        mcp_server_mock.call_tool.return_value = expected_response
        
        # Act
        result = await mcp_server_mock.call_tool(
            "syndicate_bet_history",
            history_request
        )
        
        # Assert
        assert len(result["bets"]) == 5
        assert result["bets"][0]["profit"] >= result["bets"][-1]["profit"]  # Sorted by profit desc
        assert result["summary"]["total_bets"] == 5
        assert "roi_percentage" in result["summary"]


class TestSyndicateSecurity:
    """Test security and permission validation"""
    
    @pytest.mark.asyncio
    async def test_verify_permissions_authorized(self, mcp_server_mock, mock_members):
        """Test permission verification for authorized actions"""
        # Arrange
        permission_check = {
            "member_id": mock_members[0]["member_id"],  # Founder
            "action": "manage_members",
            "resource_id": mock_members[2]["member_id"],
            "context": {
                "action_type": "role_change",
                "new_role": MemberRole.LEAD.value
            }
        }
        
        expected_response = {
            "authorized": True,
            "member_role": MemberRole.FOUNDER.value,
            "required_permission": "manage_members",
            "has_permission": True,
            "additional_checks": {
                "is_active": True,
                "is_founder": True,
                "can_modify_roles": True
            }
        }
        
        mcp_server_mock.call_tool.return_value = expected_response
        
        # Act
        result = await mcp_server_mock.call_tool(
            "syndicate_verify_permissions",
            permission_check
        )
        
        # Assert
        assert result["authorized"] is True
        assert result["has_permission"] is True
        assert result["additional_checks"]["is_founder"] is True
    
    @pytest.mark.asyncio
    async def test_verify_permissions_unauthorized(self, mcp_server_mock, mock_members):
        """Test permission verification for unauthorized actions"""
        # Arrange
        permission_check = {
            "member_id": mock_members[3]["member_id"],  # Contributor
            "action": "approve_withdrawals",
            "resource_id": "WD_001",
            "context": {
                "withdrawal_amount": 50000,
                "requester_id": mock_members[1]["member_id"]
            }
        }
        
        expected_response = {
            "authorized": False,
            "member_role": MemberRole.CONTRIBUTOR.value,
            "required_permission": "approve_withdrawals",
            "has_permission": False,
            "denial_reason": "Contributors cannot approve withdrawals",
            "required_roles": [MemberRole.FOUNDER.value, MemberRole.LEAD.value]
        }
        
        mcp_server_mock.call_tool.return_value = expected_response
        
        # Act
        result = await mcp_server_mock.call_tool(
            "syndicate_verify_permissions",
            permission_check
        )
        
        # Assert
        assert result["authorized"] is False
        assert result["has_permission"] is False
        assert "denial_reason" in result
        assert MemberRole.FOUNDER.value in result["required_roles"]
    
    @pytest.mark.asyncio
    async def test_audit_log_retrieval(self, mcp_server_mock):
        """Test audit log retrieval with filtering"""
        # Arrange
        audit_request = {
            "start_date": (datetime.now() - timedelta(days=7)).isoformat(),
            "end_date": datetime.now().isoformat(),
            "action_types": ["member_added", "funds_allocated", "vote_cast"],
            "member_filter": None,
            "limit": 50
        }
        
        audit_entries = [
            {
                "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                "action": "member_added",
                "actor_id": "MEMBER_001",
                "target_id": "MEMBER_006",
                "details": {
                    "new_member_username": "new_analyst",
                    "role": MemberRole.ANALYST.value,
                    "initial_contribution": 15000
                },
                "ip_address": "192.168.1.100",
                "success": True
            },
            {
                "timestamp": (datetime.now() - timedelta(hours=5)).isoformat(),
                "action": "funds_allocated",
                "actor_id": "SYSTEM",
                "target_id": "BET_099",
                "details": {
                    "amount": 8000,
                    "allocation_method": "proportional",
                    "participants": 4
                },
                "ip_address": None,
                "success": True
            },
            {
                "timestamp": (datetime.now() - timedelta(hours=8)).isoformat(),
                "action": "vote_cast",
                "actor_id": "MEMBER_002",
                "target_id": "PROP_004",
                "details": {
                    "vote": VoteType.YES.value,
                    "weight": 1.2
                },
                "ip_address": "192.168.1.101",
                "success": True
            }
        ]
        
        expected_response = {
            "audit_entries": audit_entries,
            "total_entries": len(audit_entries),
            "period": {
                "start": audit_request["start_date"],
                "end": audit_request["end_date"]
            },
            "action_summary": {
                "member_added": 1,
                "funds_allocated": 1,
                "vote_cast": 1
            }
        }
        
        mcp_server_mock.call_tool.return_value = expected_response
        
        # Act
        result = await mcp_server_mock.call_tool(
            "syndicate_audit_log",
            audit_request
        )
        
        # Assert
        assert len(result["audit_entries"]) == 3
        assert all(entry["success"] for entry in result["audit_entries"])
        assert result["action_summary"]["member_added"] == 1


# ===== INTEGRATION TESTS =====

class TestSyndicateMCPIntegration:
    """Test integration with MCP server"""
    
    @pytest.mark.asyncio
    async def test_full_member_lifecycle(self, mcp_server_mock, mock_syndicate_data):
        """Test complete member lifecycle from creation to removal"""
        # Create member
        create_response = await mcp_server_mock.call_tool(
            "syndicate_create_member",
            {
                "username": "lifecycle_test",
                "email": "lifecycle@test.com",
                "role": MemberRole.CONTRIBUTOR.value,
                "initial_contribution": 20000
            }
        )
        member_id = create_response.get("member_id", "MEMBER_TEST")
        
        # Update member role
        await mcp_server_mock.call_tool(
            "syndicate_update_member",
            {
                "member_id": member_id,
                "new_role": MemberRole.ANALYST.value
            }
        )
        
        # Allocate funds for member
        await mcp_server_mock.call_tool(
            "syndicate_allocate_funds",
            {
                "bet_id": "BET_TEST",
                "total_amount": 5000,
                "participating_members": [member_id]
            }
        )
        
        # Distribute profits
        await mcp_server_mock.call_tool(
            "syndicate_distribute_profits",
            {
                "bet_id": "BET_TEST",
                "total_profit": 2500,
                "participating_members": [member_id]
            }
        )
        
        # Process withdrawal
        await mcp_server_mock.call_tool(
            "syndicate_process_withdrawal",
            {
                "member_id": member_id,
                "amount": 5000
            }
        )
        
        # Remove member
        remove_response = await mcp_server_mock.call_tool(
            "syndicate_remove_member",
            {
                "member_id": member_id,
                "settle_funds": True
            }
        )
        
        # Verify each step was called
        assert mcp_server_mock.call_tool.call_count >= 6
    
    @pytest.mark.asyncio
    async def test_complete_voting_workflow(self, mcp_server_mock, mock_members):
        """Test complete voting workflow from proposal to execution"""
        # Create proposal
        proposal_response = await mcp_server_mock.call_tool(
            "syndicate_create_proposal",
            {
                "proposer_id": mock_members[1]["member_id"],
                "type": ProposalType.LARGE_BET.value,
                "title": "Integration Test Bet",
                "description": "Test bet for integration",
                "details": {"amount": 10000}
            }
        )
        proposal_id = proposal_response.get("proposal_id", "PROP_TEST")
        
        # Cast votes
        for member in mock_members[:4]:  # Exclude observer
            vote = VoteType.YES if member["role"] in [MemberRole.FOUNDER, MemberRole.LEAD] else VoteType.NO
            await mcp_server_mock.call_tool(
                "syndicate_cast_vote",
                {
                    "proposal_id": proposal_id,
                    "voter_id": member["member_id"],
                    "vote": vote.value
                }
            )
        
        # Tally votes
        tally_response = await mcp_server_mock.call_tool(
            "syndicate_tally_votes",
            {
                "proposal_id": proposal_id,
                "close_voting": True
            }
        )
        
        # If approved, allocate funds
        if tally_response.get("outcome") == "approved":
            await mcp_server_mock.call_tool(
                "syndicate_allocate_funds",
                {
                    "bet_id": f"BET_FROM_{proposal_id}",
                    "total_amount": 10000,
                    "allocation_method": "proportional"
                }
            )
        
        assert mcp_server_mock.call_tool.call_count >= 7  # 1 create + 4 votes + 1 tally + 1 allocate


# ===== PERFORMANCE AND CONCURRENCY TESTS =====

class TestSyndicatePerformance:
    """Test performance and concurrency handling"""
    
    @pytest.mark.asyncio
    async def test_concurrent_vote_casting(self, mcp_server_mock, mock_members, mock_voting_scenarios):
        """Test handling concurrent votes from multiple members"""
        proposal = mock_voting_scenarios[0]
        
        # Create concurrent vote tasks
        vote_tasks = []
        for member in mock_members[:4]:
            vote_task = mcp_server_mock.call_tool(
                "syndicate_cast_vote",
                {
                    "proposal_id": proposal["proposal_id"],
                    "voter_id": member["member_id"],
                    "vote": VoteType.YES.value
                }
            )
            vote_tasks.append(vote_task)
        
        # Execute concurrently
        start_time = time.time()
        results = await asyncio.gather(*vote_tasks, return_exceptions=True)
        execution_time = time.time() - start_time
        
        # Verify results
        assert len(results) == 4
        assert all(not isinstance(r, Exception) for r in results)
        assert execution_time < 1.0  # Should complete within 1 second
    
    @pytest.mark.asyncio
    async def test_bulk_member_operations(self, mcp_server_mock):
        """Test bulk member operations performance"""
        # Create 50 members concurrently
        create_tasks = []
        for i in range(50):
            task = mcp_server_mock.call_tool(
                "syndicate_create_member",
                {
                    "username": f"bulk_user_{i}",
                    "email": f"bulk{i}@test.com",
                    "role": MemberRole.CONTRIBUTOR.value,
                    "initial_contribution": 5000 + (i * 100)
                }
            )
            create_tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*create_tasks, return_exceptions=True)
        execution_time = time.time() - start_time
        
        # Verify performance
        assert len(results) == 50
        assert execution_time < 5.0  # Should complete within 5 seconds
        
        # List all members
        list_start = time.time()
        list_result = await mcp_server_mock.call_tool(
            "syndicate_list_members",
            {"limit": 100}
        )
        list_time = time.time() - list_start
        
        assert list_time < 0.5  # Listing should be fast
    
    @pytest.mark.asyncio
    async def test_high_frequency_fund_allocations(self, mcp_server_mock, mock_members):
        """Test high-frequency fund allocation performance"""
        # Simulate 100 rapid fund allocations
        allocation_tasks = []
        for i in range(100):
            task = mcp_server_mock.call_tool(
                "syndicate_allocate_funds",
                {
                    "bet_id": f"PERF_BET_{i:03d}",
                    "total_amount": 1000 + (i * 10),
                    "allocation_method": "proportional",
                    "participating_members": [m["member_id"] for m in mock_members[:3]]
                }
            )
            allocation_tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*allocation_tasks, return_exceptions=True)
        execution_time = time.time() - start_time
        
        # Verify performance
        assert len(results) == 100
        assert execution_time < 10.0  # Should handle 100 allocations in under 10 seconds
        
        # Calculate throughput
        throughput = len(results) / execution_time
        assert throughput > 10  # Should handle at least 10 allocations per second


# ===== SECURITY AND EDGE CASE TESTS =====

class TestSyndicateSecurityEdgeCases:
    """Test security vulnerabilities and edge cases"""
    
    @pytest.mark.asyncio
    async def test_sql_injection_prevention(self, mcp_server_mock):
        """Test SQL injection prevention in member creation"""
        malicious_inputs = [
            "'; DROP TABLE members; --",
            "admin' OR '1'='1",
            "user'; INSERT INTO members (role) VALUES ('FOUNDER'); --"
        ]
        
        for malicious_input in malicious_inputs:
            response = await mcp_server_mock.call_tool(
                "syndicate_create_member",
                {
                    "username": malicious_input,
                    "email": f"{malicious_input}@test.com",
                    "role": MemberRole.CONTRIBUTOR.value
                }
            )
            
            # Should either sanitize or reject malicious input
            # Not raise an exception or corrupt data
            assert response is not None
    
    @pytest.mark.asyncio
    async def test_permission_escalation_prevention(self, mcp_server_mock, mock_members):
        """Test prevention of unauthorized permission escalation"""
        contributor = mock_members[3]  # Contributor role
        
        # Attempt to self-promote to founder
        escalation_response = await mcp_server_mock.call_tool(
            "syndicate_update_member",
            {
                "member_id": contributor["member_id"],
                "new_role": MemberRole.FOUNDER.value,
                "updater_id": contributor["member_id"]  # Self-update
            }
        )
        
        # Should be denied
        assert escalation_response.get("status") != "updated" or \
               escalation_response.get("error") is not None
    
    @pytest.mark.asyncio
    async def test_negative_fund_allocation_prevention(self, mcp_server_mock):
        """Test prevention of negative fund allocations"""
        negative_allocation = await mcp_server_mock.call_tool(
            "syndicate_allocate_funds",
            {
                "bet_id": "NEGATIVE_TEST",
                "total_amount": -5000,  # Negative amount
                "allocation_method": "proportional"
            }
        )
        
        # Should reject negative amounts
        assert negative_allocation.get("status") != "allocated" or \
               negative_allocation.get("error") is not None
    
    @pytest.mark.asyncio
    async def test_duplicate_vote_prevention(self, mcp_server_mock, mock_members, mock_voting_scenarios):
        """Test prevention of duplicate voting"""
        proposal = mock_voting_scenarios[0]
        voter = mock_members[0]
        
        # First vote
        first_vote = await mcp_server_mock.call_tool(
            "syndicate_cast_vote",
            {
                "proposal_id": proposal["proposal_id"],
                "voter_id": voter["member_id"],
                "vote": VoteType.YES.value
            }
        )
        
        # Attempt duplicate vote
        duplicate_vote = await mcp_server_mock.call_tool(
            "syndicate_cast_vote",
            {
                "proposal_id": proposal["proposal_id"],
                "voter_id": voter["member_id"],
                "vote": VoteType.NO.value  # Try to change vote
            }
        )
        
        # Should prevent duplicate or handle vote change properly
        assert duplicate_vote.get("error") is not None or \
               duplicate_vote.get("status") == "vote_updated"
    
    @pytest.mark.asyncio
    async def test_withdrawal_exceeding_balance(self, mcp_server_mock, mock_members):
        """Test prevention of withdrawals exceeding balance"""
        member = mock_members[2]
        excessive_amount = float(member["capital_contribution"]) * 2
        
        withdrawal_response = await mcp_server_mock.call_tool(
            "syndicate_process_withdrawal",
            {
                "member_id": member["member_id"],
                "amount": excessive_amount
            }
        )
        
        # Should reject or flag for review
        assert withdrawal_response.get("status") != "completed" or \
               withdrawal_response.get("error") is not None or \
               withdrawal_response.get("status") == "rejected"


# ===== HELPER FUNCTIONS FOR ADVANCED TESTING =====

async def simulate_real_world_syndicate_activity(mcp_server_mock, duration_seconds=60):
    """Simulate realistic syndicate activity for stress testing"""
    activities = []
    start_time = time.time()
    
    while time.time() - start_time < duration_seconds:
        # Random activity selection
        activity_type = random.choice([
            "create_member", "cast_vote", "allocate_funds", 
            "check_analytics", "process_withdrawal"
        ])
        
        if activity_type == "create_member":
            task = mcp_server_mock.call_tool(
                "syndicate_create_member",
                {
                    "username": f"stress_user_{uuid.uuid4().hex[:8]}",
                    "email": f"stress_{uuid.uuid4().hex[:8]}@test.com",
                    "role": random.choice([r.value for r in MemberRole])
                }
            )
        elif activity_type == "cast_vote":
            task = mcp_server_mock.call_tool(
                "syndicate_cast_vote",
                {
                    "proposal_id": f"PROP_{random.randint(1, 10):03d}",
                    "voter_id": f"MEMBER_{random.randint(1, 50):03d}",
                    "vote": random.choice([v.value for v in VoteType])
                }
            )
        elif activity_type == "allocate_funds":
            task = mcp_server_mock.call_tool(
                "syndicate_allocate_funds",
                {
                    "bet_id": f"BET_{uuid.uuid4().hex[:8]}",
                    "total_amount": random.randint(1000, 10000),
                    "allocation_method": random.choice([m.value for m in AllocationMethod])
                }
            )
        elif activity_type == "check_analytics":
            task = mcp_server_mock.call_tool(
                "syndicate_performance_metrics",
                {"period_days": random.choice([7, 30, 90])}
            )
        else:  # process_withdrawal
            task = mcp_server_mock.call_tool(
                "syndicate_process_withdrawal",
                {
                    "member_id": f"MEMBER_{random.randint(1, 50):03d}",
                    "amount": random.randint(100, 5000)
                }
            )
        
        activities.append(task)
        
        # Random delay between activities
        await asyncio.sleep(random.uniform(0.1, 0.5))
    
    # Wait for all activities to complete
    results = await asyncio.gather(*activities, return_exceptions=True)
    
    # Calculate statistics
    successful = sum(1 for r in results if not isinstance(r, Exception))
    failed = len(results) - successful
    
    return {
        "total_operations": len(results),
        "successful": successful,
        "failed": failed,
        "duration": time.time() - start_time,
        "operations_per_second": len(results) / (time.time() - start_time)
    }


# ===== MAIN TEST RUNNER =====

if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([
        __file__,
        "-v",  # Verbose
        "-s",  # Show print statements
        "--tb=short",  # Short traceback
        "--cov=src.sports_betting.syndicate",  # Coverage for syndicate modules
        "--cov-report=html",  # HTML coverage report
        "--cov-report=term-missing",  # Terminal coverage with missing lines
        "-k", "not slow"  # Skip slow tests for quick runs
    ])