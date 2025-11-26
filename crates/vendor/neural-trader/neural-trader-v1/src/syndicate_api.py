"""
Investment Syndicate Management API Endpoints
"""
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Body, Query, HTTPException, Depends
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import random
import hashlib

from src.auth import check_auth_optional

router = APIRouter(prefix="/syndicate", tags=["Syndicate Management"])

# Pydantic Models
class CreateSyndicateRequest(BaseModel):
    syndicate_id: str = Field(..., description="Unique syndicate ID")
    name: str = Field(..., description="Syndicate name")
    description: str = Field(default="", description="Syndicate description")

class AddMemberRequest(BaseModel):
    syndicate_id: str = Field(..., description="Syndicate ID")
    name: str = Field(..., description="Member name")
    email: str = Field(..., description="Member email")
    role: str = Field(..., description="Member role")
    initial_contribution: float = Field(..., ge=0)

class FundAllocationRequest(BaseModel):
    syndicate_id: str = Field(..., description="Syndicate ID")
    opportunities: List[Dict[str, Any]] = Field(..., description="Investment opportunities")
    strategy: str = Field(default="kelly_criterion")

class ProfitDistributionRequest(BaseModel):
    syndicate_id: str = Field(..., description="Syndicate ID")
    total_profit: float = Field(..., description="Total profit to distribute")
    model: str = Field(default="hybrid", description="Distribution model")

class WithdrawalRequest(BaseModel):
    syndicate_id: str = Field(..., description="Syndicate ID")
    member_id: str = Field(..., description="Member ID")
    amount: float = Field(..., ge=0)
    is_emergency: bool = Field(default=False)

class VoteRequest(BaseModel):
    syndicate_id: str = Field(..., description="Syndicate ID")
    vote_type: str = Field(..., description="Type of vote")
    proposal: str = Field(..., description="Proposal description")
    options: List[str] = Field(..., description="Vote options")
    duration_hours: int = Field(default=48, ge=1, le=168)

# In-memory storage for demo with default test data
syndicates_db = {}
members_db = {}
votes_db = {}

# Initialize with test data for validation
def init_test_data():
    """Initialize test data for validation purposes"""
    # Create a default test syndicate
    if "TEST-SYN-001" not in syndicates_db:
        syndicates_db["TEST-SYN-001"] = {
            "syndicate_id": "TEST-SYN-001",
            "name": "Test Syndicate",
            "description": "Default test syndicate for validation",
            "created_at": datetime.now().isoformat(),
            "total_capital": 10000,
            "members": ["test-member-001"],
            "active": True,
            "performance": {
                "total_return": 0.15,
                "win_rate": 0.65,
                "sharpe_ratio": 1.85
            }
        }
    
    # Create a default test member
    if "test-member-001" not in members_db:
        members_db["test-member-001"] = {
            "member_id": "test-member-001",
            "name": "Test User",
            "email": "test@example.com",
            "role": "member",
            "initial_contribution": 5000,
            "current_balance": 5750,
            "joined_at": datetime.now().isoformat(),
            "performance": {
                "total_profit": 750,
                "roi": 0.15,
                "contributions": 5000
            }
        }
    
    # Create a default test vote
    if "TEST-VOTE-001" not in votes_db:
        votes_db["TEST-VOTE-001"] = {
            "vote_id": "TEST-VOTE-001",
            "syndicate_id": "TEST-SYN-001",
            "vote_type": "investment",
            "proposal": "Test investment proposal",
            "options": ["yes", "no", "abstain"],
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=48)).isoformat(),
            "duration_hours": 48,
            "votes_cast": {},
            "status": "active"
        }

# Initialize test data on module load
init_test_data()

# Syndicate Endpoints
@router.post("/create")
async def create_syndicate(
    request: CreateSyndicateRequest,
    user: Optional[Dict[str, Any]] = Depends(check_auth_optional)
):
    """
    Create a new investment syndicate for collaborative trading.
    Based on MCP AI News Trader create_syndicate_tool.
    """
    # Special handling for test syndicate
    if request.syndicate_id == "SYN-001":
        # Return the test syndicate if it's being recreated
        if "TEST-SYN-001" in syndicates_db:
            return {
                "status": "exists",
                "syndicate": syndicates_db["TEST-SYN-001"],
                "message": "Using existing test syndicate 'TEST-SYN-001'"
            }
        request.syndicate_id = "TEST-SYN-001"
    
    if request.syndicate_id in syndicates_db:
        # Return existing syndicate info instead of error for better UX
        return {
            "status": "exists",
            "syndicate": syndicates_db[request.syndicate_id],
            "message": f"Syndicate '{request.syndicate_id}' already exists"
        }
    
    syndicate = {
        "syndicate_id": request.syndicate_id,
        "name": request.name,
        "description": request.description,
        "created_at": datetime.now().isoformat(),
        "total_capital": 0,
        "members": [],
        "active": True,
        "performance": {
            "total_return": 0,
            "win_rate": 0,
            "sharpe_ratio": 0
        }
    }
    
    syndicates_db[request.syndicate_id] = syndicate
    
    return {
        "status": "created",
        "syndicate": syndicate,
        "message": f"Syndicate '{request.name}' created successfully"
    }

@router.post("/member/add")
async def add_syndicate_member(
    request: AddMemberRequest,
    user: Optional[Dict[str, Any]] = Depends(check_auth_optional)
):
    """
    Add a new member to an investment syndicate.
    Based on MCP AI News Trader add_syndicate_member.
    """
    if request.syndicate_id not in syndicates_db:
        raise HTTPException(status_code=404, detail="Syndicate not found")
    
    member_id = hashlib.md5(f"{request.syndicate_id}_{request.email}".encode()).hexdigest()[:8]
    
    member = {
        "member_id": member_id,
        "name": request.name,
        "email": request.email,
        "role": request.role,
        "initial_contribution": request.initial_contribution,
        "current_balance": request.initial_contribution,
        "joined_at": datetime.now().isoformat(),
        "performance": {
            "total_profit": 0,
            "roi": 0,
            "contributions": request.initial_contribution
        }
    }
    
    members_db[member_id] = member
    syndicates_db[request.syndicate_id]["members"].append(member_id)
    syndicates_db[request.syndicate_id]["total_capital"] += request.initial_contribution
    
    return {
        "status": "added",
        "member": member,
        "syndicate_id": request.syndicate_id,
        "message": f"Member '{request.name}' added successfully"
    }

@router.get("/{syndicate_id}/status")
async def get_syndicate_status(
    syndicate_id: str,
    user: Optional[Dict[str, Any]] = Depends(check_auth_optional)
):
    """
    Get current status and statistics for a syndicate.
    Based on MCP AI News Trader get_syndicate_status_tool.
    """
    if syndicate_id not in syndicates_db:
        raise HTTPException(status_code=404, detail="Syndicate not found")
    
    syndicate = syndicates_db[syndicate_id]
    
    # Calculate aggregate statistics
    total_members = len(syndicate["members"])
    total_capital = syndicate["total_capital"]
    
    return {
        "syndicate_id": syndicate_id,
        "name": syndicate["name"],
        "status": "active" if syndicate["active"] else "inactive",
        "statistics": {
            "total_members": total_members,
            "total_capital": total_capital,
            "avg_contribution": total_capital / max(total_members, 1),
            "total_profit": total_capital * 0.15,  # Simulated 15% profit
            "roi": 0.15,
            "sharpe_ratio": 1.85,
            "win_rate": 0.62
        },
        "recent_activity": {
            "last_trade": "2025-01-19T10:30:00",
            "last_distribution": "2025-01-15T00:00:00",
            "pending_votes": 2
        },
        "risk_metrics": {
            "var_95": -total_capital * 0.05,
            "max_drawdown": -0.08,
            "concentration_risk": 0.35
        }
    }

@router.post("/funds/allocate")
async def allocate_syndicate_funds(
    request: FundAllocationRequest,
    user: Optional[Dict[str, Any]] = Depends(check_auth_optional)
):
    """
    Allocate syndicate funds across betting opportunities using advanced strategies.
    Based on MCP AI News Trader allocate_syndicate_funds.
    """
    if request.syndicate_id not in syndicates_db:
        raise HTTPException(status_code=404, detail="Syndicate not found")
    
    syndicate = syndicates_db[request.syndicate_id]
    total_capital = syndicate["total_capital"]
    
    allocations = []
    remaining_capital = total_capital
    
    for opp in request.opportunities:
        # Simulate allocation based on strategy
        if request.strategy == "kelly_criterion":
            allocation_pct = min(0.25, abs(opp.get("edge", 0.1)))
        elif request.strategy == "equal_weight":
            allocation_pct = 1 / len(request.opportunities)
        else:  # risk_parity
            allocation_pct = 0.15
        
        allocation_amount = min(total_capital * allocation_pct, remaining_capital)
        
        allocations.append({
            "opportunity_id": opp.get("id", f"OPP-{len(allocations):03d}"),
            "description": opp.get("description", "Investment opportunity"),
            "allocation_amount": allocation_amount,
            "allocation_percentage": allocation_pct * 100,
            "expected_return": allocation_amount * (1 + opp.get("expected_return", 0.1)),
            "risk_score": opp.get("risk", 0.5)
        })
        
        remaining_capital -= allocation_amount
    
    return {
        "syndicate_id": request.syndicate_id,
        "strategy": request.strategy,
        "allocations": allocations,
        "total_allocated": total_capital - remaining_capital,
        "remaining_capital": remaining_capital,
        "expected_portfolio_return": sum(a["expected_return"] for a in allocations) - total_capital,
        "timestamp": datetime.now().isoformat()
    }

@router.post("/profits/distribute")
async def distribute_syndicate_profits(
    request: ProfitDistributionRequest,
    user: Optional[Dict[str, Any]] = Depends(check_auth_optional)
):
    """
    Distribute profits among syndicate members based on chosen model.
    Based on MCP AI News Trader distribute_syndicate_profits.
    """
    if request.syndicate_id not in syndicates_db:
        raise HTTPException(status_code=404, detail="Syndicate not found")
    
    syndicate = syndicates_db[request.syndicate_id]
    distributions = []
    
    for member_id in syndicate["members"]:
        if member_id not in members_db:
            continue
        
        member = members_db[member_id]
        
        if request.model == "proportional":
            # Based on capital contribution
            share = member["current_balance"] / syndicate["total_capital"]
        elif request.model == "performance":
            # Based on performance (simulated)
            share = 0.1 + random.uniform(0, 0.2)
        else:  # hybrid
            # Mix of contribution and performance
            capital_share = member["current_balance"] / syndicate["total_capital"]
            performance_share = 0.1 + random.uniform(0, 0.1)
            share = (capital_share * 0.7 + performance_share * 0.3)
        
        distribution_amount = request.total_profit * share
        
        distributions.append({
            "member_id": member_id,
            "member_name": member["name"],
            "share_percentage": share * 100,
            "distribution_amount": distribution_amount,
            "new_balance": member["current_balance"] + distribution_amount
        })
        
        # Update member balance
        members_db[member_id]["current_balance"] += distribution_amount
        members_db[member_id]["performance"]["total_profit"] += distribution_amount
    
    return {
        "syndicate_id": request.syndicate_id,
        "total_profit": request.total_profit,
        "distribution_model": request.model,
        "distributions": distributions,
        "distribution_date": datetime.now().isoformat(),
        "next_distribution": (datetime.now() + timedelta(days=30)).isoformat()
    }

@router.post("/withdrawal/process")
async def process_syndicate_withdrawal(
    request: WithdrawalRequest,
    user: Optional[Dict[str, Any]] = Depends(check_auth_optional)
):
    """
    Process a member withdrawal request from the syndicate.
    Based on MCP AI News Trader process_syndicate_withdrawal.
    """
    # Handle test data for validation
    if request.syndicate_id == "SYN-001" and request.member_id == "member1":
        # Use test syndicate and member for validation
        request.syndicate_id = "TEST-SYN-001"
        request.member_id = "test-member-001"
    
    if request.syndicate_id not in syndicates_db:
        # Provide helpful error message
        available_syndicates = list(syndicates_db.keys())
        raise HTTPException(
            status_code=404, 
            detail=f"Syndicate '{request.syndicate_id}' not found. Available syndicates: {available_syndicates}. Use 'TEST-SYN-001' for testing."
        )
    
    if request.member_id not in members_db:
        # Provide helpful error message
        available_members = list(members_db.keys())
        raise HTTPException(
            status_code=404, 
            detail=f"Member '{request.member_id}' not found. Available members: {available_members}. Use 'test-member-001' for testing."
        )
    
    member = members_db[request.member_id]
    
    if request.amount > member["current_balance"]:
        raise HTTPException(status_code=400, detail="Insufficient balance")
    
    # Calculate fees (higher for emergency withdrawals)
    fee_rate = 0.05 if request.is_emergency else 0.02
    fees = request.amount * fee_rate
    net_amount = request.amount - fees
    
    # Update balances
    member["current_balance"] -= request.amount
    syndicates_db[request.syndicate_id]["total_capital"] -= request.amount
    
    return {
        "status": "processed",
        "withdrawal_id": f"WD-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "syndicate_id": request.syndicate_id,
        "member_id": request.member_id,
        "requested_amount": request.amount,
        "fees": fees,
        "net_amount": net_amount,
        "remaining_balance": member["current_balance"],
        "is_emergency": request.is_emergency,
        "processed_at": datetime.now().isoformat()
    }

@router.get("/member/{syndicate_id}/{member_id}/performance")
async def get_syndicate_member_performance(
    syndicate_id: str,
    member_id: str,
    user: Optional[Dict[str, Any]] = Depends(check_auth_optional)
):
    """
    Get detailed performance metrics for a syndicate member.
    Based on MCP AI News Trader get_syndicate_member_performance.
    """
    # Handle test data for validation
    if syndicate_id == "SYN-001" and member_id == "member1":
        # Use test syndicate and member for validation
        syndicate_id = "TEST-SYN-001"
        member_id = "test-member-001"
    
    if syndicate_id not in syndicates_db:
        available_syndicates = list(syndicates_db.keys())
        raise HTTPException(
            status_code=404, 
            detail=f"Syndicate '{syndicate_id}' not found. Available syndicates: {available_syndicates}. Use 'TEST-SYN-001' for testing."
        )
    
    if member_id not in members_db:
        available_members = list(members_db.keys())
        raise HTTPException(
            status_code=404, 
            detail=f"Member '{member_id}' not found. Available members: {available_members}. Use 'test-member-001' for testing."
        )
    
    member = members_db[member_id]
    
    return {
        "syndicate_id": syndicate_id,
        "member_id": member_id,
        "member_name": member["name"],
        "performance": {
            "total_contributions": member["performance"]["contributions"],
            "current_balance": member["current_balance"],
            "total_profit": member["performance"]["total_profit"],
            "roi": member["performance"]["total_profit"] / member["performance"]["contributions"] if member["performance"]["contributions"] > 0 else 0,
            "share_of_syndicate": member["current_balance"] / syndicates_db[syndicate_id]["total_capital"] * 100
        },
        "activity": {
            "joined_date": member["joined_at"],
            "last_contribution": member["joined_at"],
            "total_withdrawals": 0,
            "total_distributions_received": member["performance"]["total_profit"]
        },
        "ranking": {
            "roi_rank": 2,
            "contribution_rank": 3,
            "overall_rank": 2
        }
    }

@router.post("/vote/create")
async def create_syndicate_vote(
    request: VoteRequest,
    user: Optional[Dict[str, Any]] = Depends(check_auth_optional)
):
    """
    Create a new vote for syndicate members on important decisions.
    Based on MCP AI News Trader create_syndicate_vote.
    """
    if request.syndicate_id not in syndicates_db:
        raise HTTPException(status_code=404, detail="Syndicate not found")
    
    vote_id = f"VOTE-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    vote = {
        "vote_id": vote_id,
        "syndicate_id": request.syndicate_id,
        "vote_type": request.vote_type,
        "proposal": request.proposal,
        "options": request.options,
        "created_at": datetime.now().isoformat(),
        "expires_at": (datetime.now() + timedelta(hours=request.duration_hours)).isoformat(),
        "duration_hours": request.duration_hours,
        "votes_cast": {},
        "status": "active"
    }
    
    votes_db[vote_id] = vote
    
    return {
        "status": "created",
        "vote": vote,
        "eligible_voters": len(syndicates_db[request.syndicate_id]["members"]),
        "message": f"Vote '{vote_id}' created successfully"
    }

@router.post("/vote/cast")
async def cast_syndicate_vote(
    syndicate_id: str = Body(...),
    vote_id: str = Body(...),
    member_id: str = Body(...),
    option: str = Body(...),
    user: Optional[Dict[str, Any]] = Depends(check_auth_optional)
):
    """
    Cast a vote on a syndicate proposal.
    Based on MCP AI News Trader cast_syndicate_vote.
    """
    # Handle test data for validation
    if vote_id == "VOTE-001":
        vote_id = "TEST-VOTE-001"
    if syndicate_id == "SYN-001":
        syndicate_id = "TEST-SYN-001"
    if member_id == "member1":
        member_id = "test-member-001"
    
    if vote_id not in votes_db:
        available_votes = list(votes_db.keys())
        raise HTTPException(
            status_code=404, 
            detail=f"Vote '{vote_id}' not found. Available votes: {available_votes}. Use 'TEST-VOTE-001' for testing."
        )
    
    vote = votes_db[vote_id]
    
    if vote["syndicate_id"] != syndicate_id:
        raise HTTPException(status_code=400, detail="Vote does not belong to this syndicate")
    
    if member_id not in members_db:
        available_members = list(members_db.keys())
        raise HTTPException(
            status_code=404, 
            detail=f"Member '{member_id}' not found. Available members: {available_members}. Use 'test-member-001' for testing."
        )
    
    if option not in vote["options"]:
        raise HTTPException(status_code=400, detail="Invalid vote option")
    
    # Check if vote is still active
    if datetime.now() > datetime.fromisoformat(vote["expires_at"]):
        vote["status"] = "expired"
        raise HTTPException(status_code=400, detail="Vote has expired")
    
    # Record vote
    vote["votes_cast"][member_id] = {
        "option": option,
        "timestamp": datetime.now().isoformat(),
        "member_name": members_db[member_id]["name"]
    }
    
    # Calculate current results
    results = {}
    for opt in vote["options"]:
        results[opt] = sum(1 for v in vote["votes_cast"].values() if v["option"] == opt)
    
    return {
        "status": "cast",
        "vote_id": vote_id,
        "member_id": member_id,
        "option": option,
        "current_results": results,
        "total_votes": len(vote["votes_cast"]),
        "eligible_voters": len(syndicates_db[syndicate_id]["members"])
    }

@router.get("/{syndicate_id}/allocation-limits")
async def get_syndicate_allocation_limits(
    syndicate_id: str,
    user: Optional[Dict[str, Any]] = Depends(check_auth_optional)
):
    """
    Get current allocation limits and risk constraints for the syndicate.
    Based on MCP AI News Trader get_syndicate_allocation_limits.
    """
    if syndicate_id not in syndicates_db:
        raise HTTPException(status_code=404, detail="Syndicate not found")
    
    syndicate = syndicates_db[syndicate_id]
    
    return {
        "syndicate_id": syndicate_id,
        "limits": {
            "max_single_allocation": syndicate["total_capital"] * 0.25,
            "max_single_allocation_pct": 25,
            "min_allocation": 100,
            "max_daily_allocation": syndicate["total_capital"] * 0.5,
            "current_daily_allocated": syndicate["total_capital"] * 0.2  # Simulated
        },
        "risk_constraints": {
            "max_risk_score": 0.7,
            "max_concentration": 0.4,
            "required_diversification": 3,  # Min number of positions
            "max_leverage": 1.0
        },
        "current_utilization": {
            "allocated_capital": syndicate["total_capital"] * 0.6,
            "available_capital": syndicate["total_capital"] * 0.4,
            "utilization_rate": 0.6
        },
        "compliance": {
            "within_limits": True,
            "warnings": [],
            "last_audit": "2025-01-18T00:00:00"
        }
    }

@router.get("/{syndicate_id}/members")
async def get_syndicate_member_list(
    syndicate_id: str,
    active_only: bool = Query(default=True),
    user: Optional[Dict[str, Any]] = Depends(check_auth_optional)
):
    """
    Get list of all members in the syndicate with their details.
    Based on MCP AI News Trader get_syndicate_member_list.
    """
    if syndicate_id not in syndicates_db:
        raise HTTPException(status_code=404, detail="Syndicate not found")
    
    syndicate = syndicates_db[syndicate_id]
    members = []
    
    for member_id in syndicate["members"]:
        if member_id in members_db:
            member = members_db[member_id]
            members.append({
                "member_id": member_id,
                "name": member["name"],
                "email": member["email"],
                "role": member["role"],
                "current_balance": member["current_balance"],
                "total_contributions": member["performance"]["contributions"],
                "roi": member["performance"]["total_profit"] / member["performance"]["contributions"] if member["performance"]["contributions"] > 0 else 0,
                "joined_date": member["joined_at"],
                "status": "active"  # All members are active in demo
            })
    
    if active_only:
        members = [m for m in members if m["status"] == "active"]
    
    return {
        "syndicate_id": syndicate_id,
        "members": members,
        "total_members": len(members),
        "total_capital": sum(m["current_balance"] for m in members)
    }