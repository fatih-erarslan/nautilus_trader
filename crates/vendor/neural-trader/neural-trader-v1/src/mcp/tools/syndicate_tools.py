"""
Syndicate MCP Tools Integration
Provides comprehensive syndicate management functionality through MCP protocol
"""

import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal, ROUND_DOWN
import uuid
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Import Pydantic for data validation
try:
    from pydantic import BaseModel, Field, validator
except ImportError:
    logger.error("Pydantic not installed - required for MCP tools")
    BaseModel = None

# Import syndicate modules
try:
    from ...syndicate.member_management import (
        SyndicateMemberManager, MemberRole, MemberTier, 
        MemberPerformanceTracker, VotingSystem
    )
    from ...syndicate.capital_management import (
        FundAllocationEngine, ProfitDistributionSystem,
        WithdrawalManager, AllocationStrategy, DistributionModel,
        BettingOpportunity, BankrollRules
    )
    SYNDICATE_MODULES_AVAILABLE = True
    logger.info("Syndicate modules loaded successfully")
except ImportError as e:
    logger.error(f"Failed to import syndicate modules: {e}")
    SYNDICATE_MODULES_AVAILABLE = False

# Global syndicate instances storage
SYNDICATE_INSTANCES = {}
ACTIVE_SYNDICATES = {}


# ===== Request/Response Models =====

class CreateSyndicateRequest(BaseModel):
    """Request model for creating a syndicate"""
    name: str = Field(..., description="Syndicate name")
    description: str = Field(..., description="Syndicate description")
    initial_capital: float = Field(..., description="Initial capital amount")
    bankroll_rules: Optional[Dict[str, float]] = Field(None, description="Custom bankroll rules")
    distribution_model: str = Field("hybrid", description="Profit distribution model")


class AddMemberRequest(BaseModel):
    """Request model for adding a member"""
    syndicate_id: str = Field(..., description="Syndicate ID")
    name: str = Field(..., description="Member name")
    email: str = Field(..., description="Member email")
    role: str = Field(..., description="Member role")
    initial_contribution: float = Field(..., description="Initial capital contribution")


class UpdateMemberRequest(BaseModel):
    """Request model for updating member"""
    syndicate_id: str = Field(..., description="Syndicate ID")
    member_id: str = Field(..., description="Member ID")
    new_role: Optional[str] = Field(None, description="New role")
    contribution_adjustment: Optional[float] = Field(None, description="Capital adjustment")
    is_active: Optional[bool] = Field(None, description="Active status")
    authorized_by: str = Field(..., description="ID of authorizing member")


class AllocateFundsRequest(BaseModel):
    """Request model for fund allocation"""
    syndicate_id: str = Field(..., description="Syndicate ID")
    sport: str = Field(..., description="Sport type")
    event: str = Field(..., description="Event description")
    bet_type: str = Field(..., description="Bet type")
    selection: str = Field(..., description="Selection")
    odds: float = Field(..., description="Betting odds")
    probability: float = Field(..., description="Win probability")
    edge: float = Field(..., description="Expected edge")
    confidence: float = Field(..., description="Confidence level (0-1)")
    strategy: str = Field("kelly_criterion", description="Allocation strategy")


class DistributeProfitsRequest(BaseModel):
    """Request model for profit distribution"""
    syndicate_id: str = Field(..., description="Syndicate ID")
    total_profit: float = Field(..., description="Total profit to distribute")
    distribution_model: str = Field("hybrid", description="Distribution model")
    authorized_by: str = Field(..., description="ID of authorizing member")


class CreateVoteRequest(BaseModel):
    """Request model for creating a vote"""
    syndicate_id: str = Field(..., description="Syndicate ID")
    proposal_type: str = Field(..., description="Type of proposal")
    proposal_details: Dict[str, Any] = Field(..., description="Proposal details")
    proposed_by: str = Field(..., description="ID of proposing member")
    voting_period_hours: int = Field(24, description="Voting period in hours")


class CastVoteRequest(BaseModel):
    """Request model for casting a vote"""
    syndicate_id: str = Field(..., description="Syndicate ID")
    vote_id: str = Field(..., description="Vote ID")
    member_id: str = Field(..., description="Member ID")
    decision: str = Field(..., description="Vote decision (approve/reject/abstain)")


class WithdrawalRequest(BaseModel):
    """Request model for withdrawal"""
    syndicate_id: str = Field(..., description="Syndicate ID")
    member_id: str = Field(..., description="Member ID")
    amount: float = Field(..., description="Withdrawal amount")
    is_emergency: bool = Field(False, description="Emergency withdrawal flag")


# ===== Helper Functions =====

def _get_or_create_syndicate(syndicate_id: str) -> Dict[str, Any]:
    """Get or create a syndicate instance"""
    if syndicate_id not in SYNDICATE_INSTANCES:
        # Create new instance
        SYNDICATE_INSTANCES[syndicate_id] = {
            "member_manager": SyndicateMemberManager(syndicate_id),
            "fund_allocator": None,  # Will be created with initial capital
            "profit_distributor": ProfitDistributionSystem(syndicate_id),
            "withdrawal_manager": WithdrawalManager(syndicate_id),
            "created_at": datetime.now(),
            "metadata": {}
        }
    return SYNDICATE_INSTANCES[syndicate_id]


def _validate_syndicate_exists(syndicate_id: str) -> bool:
    """Check if syndicate exists"""
    return syndicate_id in ACTIVE_SYNDICATES


def _format_decimal_response(value: Any) -> Any:
    """Format Decimal values for JSON response"""
    if isinstance(value, Decimal):
        return str(value)
    elif isinstance(value, dict):
        return {k: _format_decimal_response(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_format_decimal_response(item) for item in value]
    return value


# ===== MCP Tool Functions =====

async def syndicate_create(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new trading syndicate
    
    Example:
        params = {
            "name": "Alpha Trading Syndicate",
            "description": "Professional sports betting syndicate",
            "initial_capital": 100000.0,
            "bankroll_rules": {
                "max_single_bet": 0.05,
                "max_daily_exposure": 0.20
            },
            "distribution_model": "hybrid"
        }
    
    Returns:
        {
            "syndicate_id": "uuid-string",
            "name": "Alpha Trading Syndicate",
            "status": "active",
            "created_at": "2024-01-20T10:00:00Z",
            "initial_capital": "100000.00",
            "bankroll_rules": {...},
            "distribution_model": "hybrid"
        }
    """
    try:
        if not SYNDICATE_MODULES_AVAILABLE:
            return {"error": "Syndicate modules not available"}
        
        # Validate request
        request = CreateSyndicateRequest(**params)
        
        # Generate syndicate ID
        syndicate_id = str(uuid.uuid4())
        
        # Create syndicate instances
        instances = _get_or_create_syndicate(syndicate_id)
        
        # Initialize fund allocator with initial capital
        instances["fund_allocator"] = FundAllocationEngine(
            syndicate_id, 
            Decimal(str(request.initial_capital))
        )
        
        # Apply custom bankroll rules if provided
        if request.bankroll_rules:
            for key, value in request.bankroll_rules.items():
                if hasattr(instances["fund_allocator"].rules, key):
                    setattr(instances["fund_allocator"].rules, key, value)
        
        # Store syndicate metadata
        ACTIVE_SYNDICATES[syndicate_id] = {
            "syndicate_id": syndicate_id,
            "name": request.name,
            "description": request.description,
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "initial_capital": str(request.initial_capital),
            "current_capital": str(request.initial_capital),
            "distribution_model": request.distribution_model,
            "bankroll_rules": request.bankroll_rules or {}
        }
        
        return ACTIVE_SYNDICATES[syndicate_id]
        
    except Exception as e:
        logger.error(f"Error creating syndicate: {str(e)}")
        return {"error": str(e)}


async def syndicate_add_member(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a member to a syndicate
    
    Example:
        params = {
            "syndicate_id": "uuid-string",
            "name": "John Doe",
            "email": "john@example.com",
            "role": "senior_analyst",
            "initial_contribution": 25000.0
        }
    
    Returns:
        {
            "member_id": "uuid-string",
            "name": "John Doe",
            "role": "senior_analyst",
            "tier": "gold",
            "permissions": {...},
            "capital_contribution": "25000.00",
            "joined_date": "2024-01-20T10:00:00Z"
        }
    """
    try:
        if not SYNDICATE_MODULES_AVAILABLE:
            return {"error": "Syndicate modules not available"}
        
        # Validate request
        request = AddMemberRequest(**params)
        
        # Check syndicate exists
        if not _validate_syndicate_exists(request.syndicate_id):
            return {"error": f"Syndicate {request.syndicate_id} not found"}
        
        # Get syndicate instance
        instances = _get_or_create_syndicate(request.syndicate_id)
        member_manager = instances["member_manager"]
        
        # Convert role string to enum
        try:
            role = MemberRole(request.role)
        except ValueError:
            return {"error": f"Invalid role: {request.role}"}
        
        # Add member
        member = member_manager.add_member(
            name=request.name,
            email=request.email,
            role=role,
            initial_contribution=Decimal(str(request.initial_contribution))
        )
        
        # Update syndicate capital
        if request.syndicate_id in ACTIVE_SYNDICATES:
            current_capital = Decimal(ACTIVE_SYNDICATES[request.syndicate_id]["current_capital"])
            new_capital = current_capital + Decimal(str(request.initial_contribution))
            ACTIVE_SYNDICATES[request.syndicate_id]["current_capital"] = str(new_capital)
            
            # Update fund allocator bankroll
            if instances["fund_allocator"]:
                instances["fund_allocator"].total_bankroll = new_capital
        
        # Return member details
        return {
            "member_id": member.id,
            "name": member.name,
            "email": member.email,
            "role": member.role.value,
            "tier": member.tier.value,
            "permissions": {
                "create_syndicate": member.permissions.create_syndicate,
                "modify_strategy": member.permissions.modify_strategy,
                "approve_large_bets": member.permissions.approve_large_bets,
                "manage_members": member.permissions.manage_members,
                "distribute_profits": member.permissions.distribute_profits,
                "vote_on_strategy": member.permissions.vote_on_strategy,
                "propose_bets": member.permissions.propose_bets
            },
            "capital_contribution": str(member.capital_contribution),
            "joined_date": member.joined_date.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error adding member: {str(e)}")
        return {"error": str(e)}


async def syndicate_update_member(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update member details or status
    
    Example:
        params = {
            "syndicate_id": "uuid-string",
            "member_id": "member-uuid",
            "new_role": "lead_investor",
            "contribution_adjustment": 50000.0,
            "authorized_by": "authorizer-uuid"
        }
    
    Returns:
        {
            "member_id": "member-uuid",
            "updates_applied": {
                "role_changed": true,
                "new_role": "lead_investor",
                "contribution_updated": true,
                "new_contribution": "75000.00",
                "new_tier": "platinum"
            },
            "updated_at": "2024-01-20T10:00:00Z"
        }
    """
    try:
        if not SYNDICATE_MODULES_AVAILABLE:
            return {"error": "Syndicate modules not available"}
        
        # Validate request
        request = UpdateMemberRequest(**params)
        
        # Check syndicate exists
        if not _validate_syndicate_exists(request.syndicate_id):
            return {"error": f"Syndicate {request.syndicate_id} not found"}
        
        # Get syndicate instance
        instances = _get_or_create_syndicate(request.syndicate_id)
        member_manager = instances["member_manager"]
        
        # Check member exists
        if request.member_id not in member_manager.members:
            return {"error": f"Member {request.member_id} not found"}
        
        member = member_manager.members[request.member_id]
        updates_applied = {}
        
        # Update role if requested
        if request.new_role:
            try:
                new_role = MemberRole(request.new_role)
                member_manager.update_member_role(
                    request.member_id, 
                    new_role, 
                    request.authorized_by
                )
                updates_applied["role_changed"] = True
                updates_applied["new_role"] = request.new_role
            except (ValueError, PermissionError) as e:
                return {"error": str(e)}
        
        # Update contribution if requested
        if request.contribution_adjustment:
            old_contribution = member.capital_contribution
            new_contribution = old_contribution + Decimal(str(request.contribution_adjustment))
            
            if new_contribution < 0:
                return {"error": "Contribution cannot be negative"}
            
            member.update_tier(new_contribution)
            updates_applied["contribution_updated"] = True
            updates_applied["new_contribution"] = str(new_contribution)
            updates_applied["new_tier"] = member.tier.value
            
            # Update syndicate capital
            if request.syndicate_id in ACTIVE_SYNDICATES:
                current_capital = Decimal(ACTIVE_SYNDICATES[request.syndicate_id]["current_capital"])
                new_capital = current_capital + Decimal(str(request.contribution_adjustment))
                ACTIVE_SYNDICATES[request.syndicate_id]["current_capital"] = str(new_capital)
                
                # Update fund allocator bankroll
                if instances["fund_allocator"]:
                    instances["fund_allocator"].total_bankroll = new_capital
        
        # Update active status if requested
        if request.is_active is not None:
            if not request.is_active:
                member_manager.suspend_member(
                    request.member_id,
                    "Status update via API",
                    request.authorized_by
                )
            else:
                member.is_active = True
            updates_applied["active_status_changed"] = True
            updates_applied["is_active"] = request.is_active
        
        return {
            "member_id": request.member_id,
            "updates_applied": updates_applied,
            "updated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error updating member: {str(e)}")
        return {"error": str(e)}


async def syndicate_member_performance(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get member performance metrics
    
    Example:
        params = {
            "syndicate_id": "uuid-string",
            "member_id": "member-uuid"
        }
    
    Returns:
        {
            "member_info": {...},
            "financial_summary": {...},
            "betting_performance": {...},
            "skill_assessment": {...},
            "alpha_analysis": {...},
            "voting_weight": 0.15,
            "recent_activity": [...]
        }
    """
    try:
        if not SYNDICATE_MODULES_AVAILABLE:
            return {"error": "Syndicate modules not available"}
        
        syndicate_id = params.get("syndicate_id")
        member_id = params.get("member_id")
        
        if not syndicate_id or not member_id:
            return {"error": "Missing required parameters: syndicate_id, member_id"}
        
        # Check syndicate exists
        if not _validate_syndicate_exists(syndicate_id):
            return {"error": f"Syndicate {syndicate_id} not found"}
        
        # Get syndicate instance
        instances = _get_or_create_syndicate(syndicate_id)
        member_manager = instances["member_manager"]
        
        # Get performance report
        report = member_manager.get_member_performance_report(member_id)
        
        # Format response
        return _format_decimal_response(report)
        
    except Exception as e:
        logger.error(f"Error getting member performance: {str(e)}")
        return {"error": str(e)}


async def syndicate_list_members(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List all syndicate members
    
    Example:
        params = {
            "syndicate_id": "uuid-string",
            "include_inactive": false
        }
    
    Returns:
        {
            "total_members": 15,
            "active_members": 14,
            "members": [
                {
                    "member_id": "uuid",
                    "name": "John Doe",
                    "role": "senior_analyst",
                    "tier": "gold",
                    "capital_contribution": "25000.00",
                    "performance_score": 0.85,
                    "is_active": true
                },
                ...
            ],
            "total_capital": "500000.00"
        }
    """
    try:
        if not SYNDICATE_MODULES_AVAILABLE:
            return {"error": "Syndicate modules not available"}
        
        syndicate_id = params.get("syndicate_id")
        include_inactive = params.get("include_inactive", False)
        
        if not syndicate_id:
            return {"error": "Missing required parameter: syndicate_id"}
        
        # Check syndicate exists
        if not _validate_syndicate_exists(syndicate_id):
            return {"error": f"Syndicate {syndicate_id} not found"}
        
        # Get syndicate instance
        instances = _get_or_create_syndicate(syndicate_id)
        member_manager = instances["member_manager"]
        
        # Build member list
        members_list = []
        for member_id, member in member_manager.members.items():
            if not include_inactive and not member.is_active:
                continue
                
            members_list.append({
                "member_id": member.id,
                "name": member.name,
                "role": member.role.value,
                "tier": member.tier.value,
                "capital_contribution": str(member.capital_contribution),
                "performance_score": member.performance_score,
                "roi_score": member.roi_score,
                "accuracy_score": member.accuracy_score,
                "is_active": member.is_active,
                "joined_date": member.joined_date.isoformat()
            })
        
        # Calculate totals
        total_members = len(member_manager.members)
        active_members = sum(1 for m in member_manager.members.values() if m.is_active)
        total_capital = member_manager.get_total_capital()
        
        return {
            "total_members": total_members,
            "active_members": active_members,
            "members": members_list,
            "total_capital": str(total_capital)
        }
        
    except Exception as e:
        logger.error(f"Error listing members: {str(e)}")
        return {"error": str(e)}


async def syndicate_allocate_funds(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Allocate funds for a betting opportunity
    
    Example:
        params = {
            "syndicate_id": "uuid-string",
            "sport": "NFL",
            "event": "Chiefs vs Eagles",
            "bet_type": "moneyline",
            "selection": "Chiefs",
            "odds": 2.15,
            "probability": 0.52,
            "edge": 0.045,
            "confidence": 0.85,
            "strategy": "kelly_criterion"
        }
    
    Returns:
        {
            "allocation_id": "uuid-string",
            "amount": "2500.00",
            "percentage_of_bankroll": 0.025,
            "reasoning": {...},
            "risk_metrics": {...},
            "approval_required": false,
            "warnings": [],
            "recommended_stake_sizing": {...}
        }
    """
    try:
        if not SYNDICATE_MODULES_AVAILABLE:
            return {"error": "Syndicate modules not available"}
        
        # Validate request
        request = AllocateFundsRequest(**params)
        
        # Check syndicate exists
        if not _validate_syndicate_exists(request.syndicate_id):
            return {"error": f"Syndicate {request.syndicate_id} not found"}
        
        # Get syndicate instance
        instances = _get_or_create_syndicate(request.syndicate_id)
        fund_allocator = instances["fund_allocator"]
        
        if not fund_allocator:
            return {"error": "Fund allocator not initialized"}
        
        # Create betting opportunity
        opportunity = BettingOpportunity(
            sport=request.sport,
            event=request.event,
            bet_type=request.bet_type,
            selection=request.selection,
            odds=request.odds,
            probability=request.probability,
            edge=request.edge,
            confidence=request.confidence,
            model_agreement=0.9,  # Default high agreement
            time_until_event=timedelta(hours=2),  # Default 2 hours
            liquidity=50000.0,  # Default liquidity
            is_live=False,
            is_parlay=False
        )
        
        # Get allocation strategy
        try:
            strategy = AllocationStrategy(request.strategy)
        except ValueError:
            strategy = AllocationStrategy.KELLY_CRITERION
        
        # Allocate funds
        result = fund_allocator.allocate_funds(opportunity, strategy)
        
        # Generate allocation ID
        allocation_id = str(uuid.uuid4())
        
        # Format response
        response = {
            "allocation_id": allocation_id,
            "amount": str(result.amount),
            "percentage_of_bankroll": result.percentage_of_bankroll,
            "reasoning": result.reasoning,
            "risk_metrics": result.risk_metrics,
            "approval_required": result.approval_required,
            "warnings": result.warnings,
            "recommended_stake_sizing": _format_decimal_response(result.recommended_stake_sizing)
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error allocating funds: {str(e)}")
        return {"error": str(e)}


async def syndicate_get_exposure(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get current capital exposure
    
    Example:
        params = {
            "syndicate_id": "uuid-string"
        }
    
    Returns:
        {
            "total_bankroll": "100000.00",
            "available_capital": "75000.00",
            "current_exposure": {
                "daily": "15000.00",
                "weekly": "22000.00",
                "by_sport": {
                    "NFL": "8000.00",
                    "NBA": "7000.00"
                },
                "live_betting": "3000.00",
                "parlays": "500.00"
            },
            "exposure_percentages": {...},
            "risk_status": "moderate",
            "warnings": []
        }
    """
    try:
        if not SYNDICATE_MODULES_AVAILABLE:
            return {"error": "Syndicate modules not available"}
        
        syndicate_id = params.get("syndicate_id")
        
        if not syndicate_id:
            return {"error": "Missing required parameter: syndicate_id"}
        
        # Check syndicate exists
        if not _validate_syndicate_exists(syndicate_id):
            return {"error": f"Syndicate {syndicate_id} not found"}
        
        # Get syndicate instance
        instances = _get_or_create_syndicate(syndicate_id)
        fund_allocator = instances["fund_allocator"]
        
        if not fund_allocator:
            return {"error": "Fund allocator not initialized"}
        
        # Calculate exposures
        total_exposure = fund_allocator._calculate_total_exposure()
        available_capital = fund_allocator.total_bankroll - total_exposure
        
        # Format exposure data
        current_exposure = {
            "daily": str(fund_allocator.current_exposure["daily"]),
            "weekly": str(fund_allocator.current_exposure["weekly"]),
            "by_sport": _format_decimal_response(fund_allocator.current_exposure["by_sport"]),
            "live_betting": str(fund_allocator.current_exposure["live_betting"]),
            "parlays": str(fund_allocator.current_exposure["parlays"]),
            "open_bets": len(fund_allocator.current_exposure["open_bets"])
        }
        
        # Calculate percentages
        exposure_percentages = {
            "daily": float(fund_allocator.current_exposure["daily"] / fund_allocator.total_bankroll),
            "weekly": float(fund_allocator.current_exposure["weekly"] / fund_allocator.total_bankroll),
            "total": float(total_exposure / fund_allocator.total_bankroll)
        }
        
        # Determine risk status
        if exposure_percentages["total"] > 0.5:
            risk_status = "high"
        elif exposure_percentages["total"] > 0.3:
            risk_status = "moderate"
        else:
            risk_status = "low"
        
        # Generate warnings
        warnings = []
        if exposure_percentages["daily"] > fund_allocator.rules.max_daily_exposure * 0.8:
            warnings.append("Approaching daily exposure limit")
        
        if available_capital < fund_allocator.total_bankroll * Decimal(str(fund_allocator.rules.minimum_reserve)):
            warnings.append("Below minimum reserve threshold")
        
        return {
            "total_bankroll": str(fund_allocator.total_bankroll),
            "available_capital": str(available_capital),
            "current_exposure": current_exposure,
            "exposure_percentages": exposure_percentages,
            "risk_status": risk_status,
            "warnings": warnings
        }
        
    except Exception as e:
        logger.error(f"Error getting exposure: {str(e)}")
        return {"error": str(e)}


async def syndicate_distribute_profits(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Distribute profits to members
    
    Example:
        params = {
            "syndicate_id": "uuid-string",
            "total_profit": 50000.0,
            "distribution_model": "hybrid",
            "authorized_by": "authorizer-uuid"
        }
    
    Returns:
        {
            "distribution_id": "uuid-string",
            "total_profit": "50000.00",
            "operational_reserve": "2500.00",
            "distributed_amount": "47500.00",
            "distributions": {
                "member-uuid": {
                    "gross_amount": "10000.00",
                    "tax_withheld": "2400.00",
                    "net_amount": "7600.00",
                    "payment_method": "bank_transfer"
                },
                ...
            },
            "distribution_date": "2024-01-20T10:00:00Z"
        }
    """
    try:
        if not SYNDICATE_MODULES_AVAILABLE:
            return {"error": "Syndicate modules not available"}
        
        # Validate request
        request = DistributeProfitsRequest(**params)
        
        # Check syndicate exists
        if not _validate_syndicate_exists(request.syndicate_id):
            return {"error": f"Syndicate {request.syndicate_id} not found"}
        
        # Get syndicate instance
        instances = _get_or_create_syndicate(request.syndicate_id)
        member_manager = instances["member_manager"]
        profit_distributor = instances["profit_distributor"]
        
        # Check authorization
        authorizer = member_manager.members.get(request.authorized_by)
        if not authorizer or not authorizer.permissions.distribute_profits:
            return {"error": f"Member {request.authorized_by} not authorized to distribute profits"}
        
        # Prepare member data for distribution
        members_data = []
        for member_id, member in member_manager.members.items():
            if member.is_active:
                members_data.append({
                    "id": member_id,
                    "capital_contribution": str(member.capital_contribution),
                    "performance_score": member.performance_score,
                    "roi_score": member.roi_score,
                    "win_rate": member.stats.win_rate if member.stats else 0.0,
                    "consistency_score": 0.7,  # Default consistency
                    "tier": member.tier.value,
                    "is_active": member.is_active,
                    "tax_jurisdiction": "US",  # Default jurisdiction
                    "payment_method": "bank_transfer"
                })
        
        # Get distribution model
        try:
            model = DistributionModel(request.distribution_model)
        except ValueError:
            model = DistributionModel.HYBRID
        
        # Calculate distribution
        distributions = profit_distributor.calculate_distribution(
            Decimal(str(request.total_profit)),
            members_data,
            model
        )
        
        # Generate distribution ID
        distribution_id = str(uuid.uuid4())
        
        # Calculate totals
        operational_reserve = Decimal(str(request.total_profit)) * Decimal("0.05")
        distributed_amount = Decimal(str(request.total_profit)) - operational_reserve
        
        return {
            "distribution_id": distribution_id,
            "total_profit": str(request.total_profit),
            "operational_reserve": str(operational_reserve),
            "distributed_amount": str(distributed_amount),
            "distributions": distributions,
            "distribution_model": request.distribution_model,
            "distribution_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error distributing profits: {str(e)}")
        return {"error": str(e)}


async def syndicate_process_withdrawal(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process member withdrawal request
    
    Example:
        params = {
            "syndicate_id": "uuid-string",
            "member_id": "member-uuid",
            "amount": 10000.0,
            "is_emergency": false
        }
    
    Returns:
        {
            "withdrawal_id": "uuid-string",
            "member_id": "member-uuid",
            "requested_amount": "10000.00",
            "approved_amount": "10000.00",
            "penalty": "0.00",
            "net_amount": "10000.00",
            "status": "scheduled",
            "scheduled_for": "2024-01-27T10:00:00Z",
            "voting_power_impact": {...}
        }
    """
    try:
        if not SYNDICATE_MODULES_AVAILABLE:
            return {"error": "Syndicate modules not available"}
        
        # Validate request
        request = WithdrawalRequest(**params)
        
        # Check syndicate exists
        if not _validate_syndicate_exists(request.syndicate_id):
            return {"error": f"Syndicate {request.syndicate_id} not found"}
        
        # Get syndicate instance
        instances = _get_or_create_syndicate(request.syndicate_id)
        member_manager = instances["member_manager"]
        withdrawal_manager = instances["withdrawal_manager"]
        
        # Check member exists
        if request.member_id not in member_manager.members:
            return {"error": f"Member {request.member_id} not found"}
        
        member = member_manager.members[request.member_id]
        
        # Process withdrawal request
        result = withdrawal_manager.request_withdrawal(
            member_id=request.member_id,
            member_balance=member.capital_contribution,
            amount=Decimal(str(request.amount)),
            is_emergency=request.is_emergency
        )
        
        # Update member's capital if approved
        if result.get("status") in ["scheduled", "partial_approval"]:
            approved_amount = Decimal(result.get("approved_amount", result.get("net_amount", "0")))
            new_contribution = member.capital_contribution - approved_amount
            member.update_tier(new_contribution)
            
            # Update syndicate capital
            if request.syndicate_id in ACTIVE_SYNDICATES:
                current_capital = Decimal(ACTIVE_SYNDICATES[request.syndicate_id]["current_capital"])
                new_capital = current_capital - approved_amount
                ACTIVE_SYNDICATES[request.syndicate_id]["current_capital"] = str(new_capital)
                
                # Update fund allocator bankroll
                if instances["fund_allocator"]:
                    instances["fund_allocator"].total_bankroll = new_capital
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing withdrawal: {str(e)}")
        return {"error": str(e)}


async def syndicate_get_balance(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get syndicate balance and financial status
    
    Example:
        params = {
            "syndicate_id": "uuid-string"
        }
    
    Returns:
        {
            "syndicate_id": "uuid-string",
            "total_capital": "100000.00",
            "available_capital": "75000.00",
            "reserved_capital": "25000.00",
            "member_contributions": "95000.00",
            "retained_profits": "5000.00",
            "pending_withdrawals": "2000.00",
            "financial_health": "excellent"
        }
    """
    try:
        if not SYNDICATE_MODULES_AVAILABLE:
            return {"error": "Syndicate modules not available"}
        
        syndicate_id = params.get("syndicate_id")
        
        if not syndicate_id:
            return {"error": "Missing required parameter: syndicate_id"}
        
        # Check syndicate exists
        if not _validate_syndicate_exists(syndicate_id):
            return {"error": f"Syndicate {syndicate_id} not found"}
        
        # Get syndicate instance
        instances = _get_or_create_syndicate(syndicate_id)
        member_manager = instances["member_manager"]
        fund_allocator = instances["fund_allocator"]
        withdrawal_manager = instances["withdrawal_manager"]
        
        # Calculate various capital components
        total_capital = member_manager.get_total_capital()
        
        if fund_allocator:
            total_exposure = fund_allocator._calculate_total_exposure()
            available_capital = fund_allocator.total_bankroll - total_exposure
            reserved_capital = fund_allocator.total_bankroll * Decimal(str(fund_allocator.rules.minimum_reserve))
        else:
            available_capital = total_capital
            reserved_capital = Decimal("0")
            total_exposure = Decimal("0")
        
        # Calculate member contributions
        member_contributions = sum(
            member.capital_contribution 
            for member in member_manager.members.values() 
            if member.is_active
        )
        
        # Calculate retained profits
        retained_profits = total_capital - member_contributions
        
        # Calculate pending withdrawals
        pending_withdrawals = Decimal("0")
        for withdrawal in withdrawal_manager.withdrawal_requests:
            if withdrawal["status"] == "scheduled":
                pending_withdrawals += Decimal(withdrawal["net_amount"])
        
        # Determine financial health
        if available_capital > total_capital * Decimal("0.5"):
            financial_health = "excellent"
        elif available_capital > total_capital * Decimal("0.3"):
            financial_health = "good"
        elif available_capital > total_capital * Decimal("0.1"):
            financial_health = "fair"
        else:
            financial_health = "poor"
        
        return {
            "syndicate_id": syndicate_id,
            "total_capital": str(total_capital),
            "available_capital": str(available_capital),
            "reserved_capital": str(reserved_capital),
            "current_exposure": str(total_exposure),
            "member_contributions": str(member_contributions),
            "retained_profits": str(retained_profits),
            "pending_withdrawals": str(pending_withdrawals),
            "financial_health": financial_health
        }
        
    except Exception as e:
        logger.error(f"Error getting balance: {str(e)}")
        return {"error": str(e)}


async def syndicate_create_vote(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a governance vote
    
    Example:
        params = {
            "syndicate_id": "uuid-string",
            "proposal_type": "strategy_change",
            "proposal_details": {
                "title": "Increase max bet size",
                "description": "Propose to increase max single bet from 5% to 7%",
                "changes": {"max_single_bet": 0.07}
            },
            "proposed_by": "member-uuid",
            "voting_period_hours": 48
        }
    
    Returns:
        {
            "vote_id": "uuid-string",
            "proposal_type": "strategy_change",
            "proposal_details": {...},
            "proposed_by": "member-uuid",
            "created_at": "2024-01-20T10:00:00Z",
            "expires_at": "2024-01-22T10:00:00Z",
            "status": "active"
        }
    """
    try:
        if not SYNDICATE_MODULES_AVAILABLE:
            return {"error": "Syndicate modules not available"}
        
        # Validate request
        request = CreateVoteRequest(**params)
        
        # Check syndicate exists
        if not _validate_syndicate_exists(request.syndicate_id):
            return {"error": f"Syndicate {request.syndicate_id} not found"}
        
        # Get syndicate instance
        instances = _get_or_create_syndicate(request.syndicate_id)
        member_manager = instances["member_manager"]
        voting_system = member_manager.voting_system
        
        # Check if proposer has permission
        proposer = member_manager.members.get(request.proposed_by)
        if not proposer:
            return {"error": f"Member {request.proposed_by} not found"}
        
        # Check proposal type permissions
        if request.proposal_type == "strategy_change" and not proposer.permissions.vote_on_strategy:
            return {"error": "Member not authorized to propose strategy changes"}
        
        # Create vote
        vote_id = voting_system.create_vote(
            proposal_type=request.proposal_type,
            proposal_details=request.proposal_details,
            proposed_by=request.proposed_by,
            voting_period_hours=request.voting_period_hours
        )
        
        # Get vote details
        vote = voting_system.active_votes[vote_id]
        
        return {
            "vote_id": vote_id,
            "proposal_type": vote["type"],
            "proposal_details": vote["details"],
            "proposed_by": vote["proposed_by"],
            "created_at": vote["created_at"].isoformat(),
            "expires_at": vote["expires_at"].isoformat(),
            "status": vote["status"]
        }
        
    except Exception as e:
        logger.error(f"Error creating vote: {str(e)}")
        return {"error": str(e)}


async def syndicate_cast_vote(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cast a vote on a proposal
    
    Example:
        params = {
            "syndicate_id": "uuid-string",
            "vote_id": "vote-uuid",
            "member_id": "member-uuid",
            "decision": "approve"
        }
    
    Returns:
        {
            "vote_id": "vote-uuid",
            "member_id": "member-uuid",
            "decision": "approve",
            "voting_weight": 0.15,
            "vote_recorded": true,
            "timestamp": "2024-01-20T10:00:00Z"
        }
    """
    try:
        if not SYNDICATE_MODULES_AVAILABLE:
            return {"error": "Syndicate modules not available"}
        
        # Validate request
        request = CastVoteRequest(**params)
        
        # Check syndicate exists
        if not _validate_syndicate_exists(request.syndicate_id):
            return {"error": f"Syndicate {request.syndicate_id} not found"}
        
        # Get syndicate instance
        instances = _get_or_create_syndicate(request.syndicate_id)
        member_manager = instances["member_manager"]
        voting_system = member_manager.voting_system
        
        # Validate decision
        if request.decision not in ["approve", "reject", "abstain"]:
            return {"error": "Invalid decision. Must be: approve, reject, or abstain"}
        
        # Cast vote
        try:
            success = voting_system.cast_vote(
                vote_id=request.vote_id,
                member_id=request.member_id,
                decision=request.decision
            )
            
            if success:
                # Get vote details
                vote = voting_system.active_votes[request.vote_id]
                member_vote = vote["votes"][request.member_id]
                
                return {
                    "vote_id": request.vote_id,
                    "member_id": request.member_id,
                    "decision": request.decision,
                    "voting_weight": member_vote["weight"],
                    "vote_recorded": True,
                    "timestamp": member_vote["timestamp"].isoformat()
                }
            else:
                return {"error": "Failed to record vote"}
                
        except ValueError as e:
            return {"error": str(e)}
            
    except Exception as e:
        logger.error(f"Error casting vote: {str(e)}")
        return {"error": str(e)}


async def syndicate_vote_results(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get voting results
    
    Example:
        params = {
            "syndicate_id": "uuid-string",
            "vote_id": "vote-uuid"
        }
    
    Returns:
        {
            "vote_id": "vote-uuid",
            "status": "active",
            "results": {
                "approve": 0.65,
                "reject": 0.25,
                "abstain": 0.10
            },
            "total_votes": 12,
            "total_weight": 0.87,
            "approval_percentage": 65.0,
            "participation_rate": 80.0,
            "outcome": "approved"
        }
    """
    try:
        if not SYNDICATE_MODULES_AVAILABLE:
            return {"error": "Syndicate modules not available"}
        
        syndicate_id = params.get("syndicate_id")
        vote_id = params.get("vote_id")
        
        if not syndicate_id or not vote_id:
            return {"error": "Missing required parameters: syndicate_id, vote_id"}
        
        # Check syndicate exists
        if not _validate_syndicate_exists(syndicate_id):
            return {"error": f"Syndicate {syndicate_id} not found"}
        
        # Get syndicate instance
        instances = _get_or_create_syndicate(syndicate_id)
        member_manager = instances["member_manager"]
        voting_system = member_manager.voting_system
        
        # Get results
        try:
            results = voting_system.get_vote_results(vote_id)
            
            # Determine outcome
            if results["status"] == "active":
                outcome = "pending"
            elif results["approval_percentage"] >= 50.0:
                outcome = "approved"
            else:
                outcome = "rejected"
            
            results["outcome"] = outcome
            
            return results
            
        except ValueError as e:
            return {"error": str(e)}
            
    except Exception as e:
        logger.error(f"Error getting vote results: {str(e)}")
        return {"error": str(e)}


async def syndicate_get_rules(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get syndicate governance rules
    
    Example:
        params = {
            "syndicate_id": "uuid-string"
        }
    
    Returns:
        {
            "syndicate_id": "uuid-string",
            "bankroll_rules": {...},
            "voting_rules": {...},
            "withdrawal_rules": {...},
            "distribution_rules": {...},
            "membership_rules": {...}
        }
    """
    try:
        if not SYNDICATE_MODULES_AVAILABLE:
            return {"error": "Syndicate modules not available"}
        
        syndicate_id = params.get("syndicate_id")
        
        if not syndicate_id:
            return {"error": "Missing required parameter: syndicate_id"}
        
        # Check syndicate exists
        if not _validate_syndicate_exists(syndicate_id):
            return {"error": f"Syndicate {syndicate_id} not found"}
        
        # Get syndicate instance
        instances = _get_or_create_syndicate(syndicate_id)
        fund_allocator = instances["fund_allocator"]
        withdrawal_manager = instances["withdrawal_manager"]
        
        # Compile rules
        rules = {
            "syndicate_id": syndicate_id,
            "bankroll_rules": {
                "max_single_bet": fund_allocator.rules.max_single_bet if fund_allocator else 0.05,
                "max_daily_exposure": fund_allocator.rules.max_daily_exposure if fund_allocator else 0.20,
                "max_sport_concentration": fund_allocator.rules.max_sport_concentration if fund_allocator else 0.40,
                "minimum_reserve": fund_allocator.rules.minimum_reserve if fund_allocator else 0.30,
                "stop_loss_daily": fund_allocator.rules.stop_loss_daily if fund_allocator else 0.10,
                "stop_loss_weekly": fund_allocator.rules.stop_loss_weekly if fund_allocator else 0.20,
                "profit_lock": fund_allocator.rules.profit_lock if fund_allocator else 0.50,
                "max_parlay_percentage": fund_allocator.rules.max_parlay_percentage if fund_allocator else 0.02,
                "max_live_betting": fund_allocator.rules.max_live_betting if fund_allocator else 0.15
            },
            "voting_rules": {
                "default_voting_period_hours": 24,
                "emergency_voting_period_hours": 6,
                "approval_threshold": 0.50,
                "quorum_requirement": 0.33,
                "veto_threshold": 0.75
            },
            "withdrawal_rules": withdrawal_manager.rules,
            "distribution_rules": {
                "operational_reserve_percentage": 0.05,
                "distribution_frequency": "monthly",
                "minimum_profit_for_distribution": 1000.0,
                "tax_withholding_enabled": True
            },
            "membership_rules": {
                "minimum_contribution": 1000.0,
                "maximum_members": 50,
                "lockup_period_days": 90,
                "performance_review_period": "quarterly",
                "inactive_member_threshold_days": 180
            }
        }
        
        # Add syndicate-specific overrides if any
        if syndicate_id in ACTIVE_SYNDICATES:
            syndicate_data = ACTIVE_SYNDICATES[syndicate_id]
            if "bankroll_rules" in syndicate_data:
                rules["bankroll_rules"].update(syndicate_data["bankroll_rules"])
        
        return rules
        
    except Exception as e:
        logger.error(f"Error getting rules: {str(e)}")
        return {"error": str(e)}


async def syndicate_performance_report(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate comprehensive performance report
    
    Example:
        params = {
            "syndicate_id": "uuid-string",
            "period_days": 30
        }
    
    Returns:
        {
            "syndicate_id": "uuid-string",
            "report_period": {
                "start_date": "2023-12-21T00:00:00Z",
                "end_date": "2024-01-20T23:59:59Z",
                "days": 30
            },
            "financial_performance": {...},
            "betting_statistics": {...},
            "member_rankings": [...],
            "allocation_efficiency": {...},
            "risk_metrics": {...}
        }
    """
    try:
        if not SYNDICATE_MODULES_AVAILABLE:
            return {"error": "Syndicate modules not available"}
        
        syndicate_id = params.get("syndicate_id")
        period_days = params.get("period_days", 30)
        
        if not syndicate_id:
            return {"error": "Missing required parameter: syndicate_id"}
        
        # Check syndicate exists
        if not _validate_syndicate_exists(syndicate_id):
            return {"error": f"Syndicate {syndicate_id} not found"}
        
        # Get syndicate instance
        instances = _get_or_create_syndicate(syndicate_id)
        member_manager = instances["member_manager"]
        fund_allocator = instances["fund_allocator"]
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        # Mock performance data (in production, would aggregate from historical data)
        total_capital = member_manager.get_total_capital()
        
        # Financial performance
        financial_performance = {
            "starting_capital": str(total_capital * Decimal("0.9")),  # Mock 10% growth
            "ending_capital": str(total_capital),
            "total_return": "11.11",  # 10% growth = 11.11% return
            "total_profit": str(total_capital * Decimal("0.1")),
            "total_staked": str(total_capital * Decimal("2.5")),  # Mock turnover
            "roi": "4.0",  # 10% profit on 250% turnover
            "sharpe_ratio": 1.85,
            "max_drawdown": -0.08,
            "win_rate": 0.545
        }
        
        # Betting statistics
        betting_statistics = {
            "total_bets_placed": 156,
            "winning_bets": 85,
            "losing_bets": 71,
            "average_odds": 2.15,
            "average_stake": str(total_capital * Decimal("0.016")),
            "biggest_win": str(total_capital * Decimal("0.05")),
            "biggest_loss": str(total_capital * Decimal("0.03")),
            "by_sport": {
                "NFL": {"bets": 45, "win_rate": 0.556, "roi": 4.2},
                "NBA": {"bets": 38, "win_rate": 0.526, "roi": 3.1},
                "MLB": {"bets": 29, "win_rate": 0.517, "roi": 2.8},
                "Soccer": {"bets": 44, "win_rate": 0.568, "roi": 5.5}
            }
        }
        
        # Member rankings
        member_rankings = []
        for i, (member_id, member) in enumerate(member_manager.members.items()):
            if member.is_active:
                member_rankings.append({
                    "rank": i + 1,
                    "member_id": member_id,
                    "name": member.name,
                    "roi": member.stats.roi if member.stats else 0.0,
                    "win_rate": member.stats.win_rate if member.stats else 0.0,
                    "profit_contribution": str(member.stats.total_profit if member.stats else 0),
                    "alpha": 0.02 + (i * 0.001)  # Mock alpha
                })
        
        member_rankings.sort(key=lambda x: x["roi"], reverse=True)
        
        # Allocation efficiency
        allocation_efficiency = {
            "kelly_criterion_adherence": 0.92,
            "average_allocation_percentage": 0.016,
            "risk_adjusted_returns": 1.85,
            "allocation_accuracy": 0.88,
            "timing_efficiency": 0.79
        }
        
        # Risk metrics
        risk_metrics = {
            "value_at_risk_95": float(total_capital) * 0.05,
            "conditional_value_at_risk": float(total_capital) * 0.08,
            "beta": 0.65,
            "correlation_with_market": 0.12,
            "downside_deviation": 0.045,
            "maximum_daily_loss": float(total_capital) * 0.032,
            "recovery_time_days": 3.5
        }
        
        return {
            "syndicate_id": syndicate_id,
            "report_period": {
                "start_date": start_date.isoformat() + "Z",
                "end_date": end_date.isoformat() + "Z",
                "days": period_days
            },
            "financial_performance": financial_performance,
            "betting_statistics": betting_statistics,
            "member_rankings": member_rankings[:10],  # Top 10
            "allocation_efficiency": allocation_efficiency,
            "risk_metrics": risk_metrics
        }
        
    except Exception as e:
        logger.error(f"Error generating performance report: {str(e)}")
        return {"error": str(e)}


async def syndicate_risk_analysis(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform risk analysis
    
    Example:
        params = {
            "syndicate_id": "uuid-string",
            "include_scenarios": true
        }
    
    Returns:
        {
            "syndicate_id": "uuid-string",
            "current_risk_profile": {...},
            "risk_concentrations": {...},
            "stress_test_results": {...},
            "risk_recommendations": [...],
            "risk_score": 72
        }
    """
    try:
        if not SYNDICATE_MODULES_AVAILABLE:
            return {"error": "Syndicate modules not available"}
        
        syndicate_id = params.get("syndicate_id")
        include_scenarios = params.get("include_scenarios", True)
        
        if not syndicate_id:
            return {"error": "Missing required parameter: syndicate_id"}
        
        # Check syndicate exists
        if not _validate_syndicate_exists(syndicate_id):
            return {"error": f"Syndicate {syndicate_id} not found"}
        
        # Get syndicate instance
        instances = _get_or_create_syndicate(syndicate_id)
        fund_allocator = instances["fund_allocator"]
        member_manager = instances["member_manager"]
        
        if not fund_allocator:
            return {"error": "Fund allocator not initialized"}
        
        # Current risk profile
        total_exposure = fund_allocator._calculate_total_exposure()
        exposure_percentage = float(total_exposure / fund_allocator.total_bankroll) if fund_allocator.total_bankroll > 0 else 0
        
        current_risk_profile = {
            "total_exposure": str(total_exposure),
            "exposure_percentage": exposure_percentage,
            "daily_exposure": str(fund_allocator.current_exposure["daily"]),
            "live_betting_exposure": str(fund_allocator.current_exposure["live_betting"]),
            "parlay_exposure": str(fund_allocator.current_exposure["parlays"]),
            "open_positions": len(fund_allocator.current_exposure["open_bets"]),
            "leverage_ratio": 1.0  # No leverage in sports betting
        }
        
        # Risk concentrations
        risk_concentrations = {
            "by_sport": _format_decimal_response(fund_allocator.current_exposure["by_sport"]),
            "largest_single_bet": str(max([Decimal("0")] + [bet["amount"] for bet in fund_allocator.current_exposure["open_bets"]])),
            "top_3_bets_percentage": 0.0,  # Would calculate from actual bets
            "member_concentration": {},  # Would calculate member exposure concentration
            "time_concentration": "moderate"  # Based on event timing
        }
        
        # Calculate top member concentrations
        total_capital = member_manager.get_total_capital()
        for member_id, member in member_manager.members.items():
            if member.is_active and total_capital > 0:
                concentration = float(member.capital_contribution / total_capital)
                if concentration > 0.1:  # Members with >10% stake
                    risk_concentrations["member_concentration"][member.name] = concentration
        
        # Stress test results
        stress_test_results = {}
        if include_scenarios:
            stress_test_results = {
                "scenarios": {
                    "10_loss_streak": {
                        "probability": 0.001,
                        "capital_impact": -0.15,
                        "recovery_time_days": 45
                    },
                    "major_upset_weekend": {
                        "probability": 0.05,
                        "capital_impact": -0.08,
                        "recovery_time_days": 14
                    },
                    "black_swan_event": {
                        "probability": 0.0001,
                        "capital_impact": -0.25,
                        "recovery_time_days": 90
                    }
                },
                "var_95": float(total_capital) * 0.05,
                "cvar_95": float(total_capital) * 0.08,
                "maximum_tolerable_loss": float(total_capital) * 0.30
            }
        
        # Risk recommendations
        risk_recommendations = []
        
        if exposure_percentage > 0.15:
            risk_recommendations.append({
                "priority": "high",
                "category": "exposure",
                "recommendation": "Reduce overall exposure below 15% of bankroll",
                "impact": "Reduce risk of significant drawdown"
            })
        
        if len(risk_concentrations["by_sport"]) < 3:
            risk_recommendations.append({
                "priority": "medium",
                "category": "diversification",
                "recommendation": "Diversify across more sports",
                "impact": "Reduce sport-specific risk"
            })
        
        if fund_allocator.current_exposure["parlays"] > fund_allocator.total_bankroll * Decimal("0.01"):
            risk_recommendations.append({
                "priority": "medium",
                "category": "bet_type",
                "recommendation": "Reduce parlay exposure",
                "impact": "Parlays have higher variance"
            })
        
        # Calculate risk score (0-100, higher is better)
        risk_score = 100
        risk_score -= exposure_percentage * 100  # Deduct for exposure
        risk_score -= len(risk_concentrations["by_sport"]) < 3 and 10 or 0  # Deduct for low diversification
        risk_score -= float(fund_allocator.current_exposure["live_betting"] / fund_allocator.total_bankroll) * 50
        risk_score = max(0, min(100, risk_score))
        
        return {
            "syndicate_id": syndicate_id,
            "current_risk_profile": current_risk_profile,
            "risk_concentrations": risk_concentrations,
            "stress_test_results": stress_test_results,
            "risk_recommendations": risk_recommendations,
            "risk_score": int(risk_score)
        }
        
    except Exception as e:
        logger.error(f"Error performing risk analysis: {str(e)}")
        return {"error": str(e)}


async def syndicate_tax_report(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate tax report for syndicate
    
    Example:
        params = {
            "syndicate_id": "uuid-string",
            "tax_year": 2024,
            "include_member_details": true
        }
    
    Returns:
        {
            "syndicate_id": "uuid-string",
            "tax_year": 2024,
            "summary": {...},
            "distributions": {...},
            "member_1099s": [...],
            "tax_obligations": {...},
            "filing_requirements": [...]
        }
    """
    try:
        if not SYNDICATE_MODULES_AVAILABLE:
            return {"error": "Syndicate modules not available"}
        
        syndicate_id = params.get("syndicate_id")
        tax_year = params.get("tax_year", datetime.now().year)
        include_member_details = params.get("include_member_details", True)
        
        if not syndicate_id:
            return {"error": "Missing required parameter: syndicate_id"}
        
        # Check syndicate exists
        if not _validate_syndicate_exists(syndicate_id):
            return {"error": f"Syndicate {syndicate_id} not found"}
        
        # Get syndicate instance
        instances = _get_or_create_syndicate(syndicate_id)
        member_manager = instances["member_manager"]
        profit_distributor = instances["profit_distributor"]
        
        # Mock tax data (in production, would aggregate from actual transactions)
        total_revenue = 250000.0
        total_expenses = 5000.0
        total_profit = total_revenue - total_expenses
        total_distributions = total_profit * 0.8  # 80% distributed
        
        # Summary
        summary = {
            "gross_revenue": total_revenue,
            "deductible_expenses": total_expenses,
            "net_profit": total_profit,
            "total_distributions": total_distributions,
            "retained_earnings": total_profit - total_distributions,
            "effective_tax_rate": 0.24  # Estimated
        }
        
        # Distributions summary
        distributions_summary = {
            "total_distributed": total_distributions,
            "distribution_count": 12,  # Monthly
            "average_distribution": total_distributions / 12,
            "by_quarter": {
                "Q1": total_distributions * 0.22,
                "Q2": total_distributions * 0.28,
                "Q3": total_distributions * 0.25,
                "Q4": total_distributions * 0.25
            }
        }
        
        # Member 1099s
        member_1099s = []
        if include_member_details:
            for member_id, member in member_manager.members.items():
                if member.is_active:
                    # Calculate member's share
                    member_share = float(member.capital_contribution / member_manager.get_total_capital())
                    member_distributions = total_distributions * member_share
                    
                    # Tax withholding (simplified)
                    tax_rate = 0.24  # Federal rate
                    tax_withheld = member_distributions * tax_rate
                    
                    member_1099s.append({
                        "member_id": member_id,
                        "name": member.name,
                        "tax_id": "XXX-XX-" + member_id[-4:],  # Masked
                        "total_distributions": member_distributions,
                        "federal_tax_withheld": tax_withheld,
                        "state_tax_withheld": member_distributions * 0.05,  # Example state
                        "form_type": "1099-MISC",
                        "box_7_nonemployee_compensation": member_distributions
                    })
        
        # Tax obligations
        tax_obligations = {
            "federal": {
                "estimated_tax": total_profit * 0.21,  # Corporate rate
                "quarterly_payments": [
                    {"quarter": "Q1", "due_date": f"{tax_year}-04-15", "amount": total_profit * 0.21 * 0.25},
                    {"quarter": "Q2", "due_date": f"{tax_year}-06-15", "amount": total_profit * 0.21 * 0.25},
                    {"quarter": "Q3", "due_date": f"{tax_year}-09-15", "amount": total_profit * 0.21 * 0.25},
                    {"quarter": "Q4", "due_date": f"{tax_year + 1}-01-15", "amount": total_profit * 0.21 * 0.25}
                ]
            },
            "state": {
                "estimated_tax": total_profit * 0.05,
                "jurisdiction": "Example State"
            },
            "total_estimated_tax": total_profit * 0.26
        }
        
        # Filing requirements
        filing_requirements = [
            {
                "form": "1120",
                "description": "U.S. Corporation Income Tax Return",
                "due_date": f"{tax_year + 1}-03-15",
                "status": "required"
            },
            {
                "form": "1099-MISC",
                "description": "Miscellaneous Income for each member",
                "due_date": f"{tax_year + 1}-01-31",
                "status": "required",
                "count": len(member_1099s)
            },
            {
                "form": "1096",
                "description": "Annual Summary and Transmittal",
                "due_date": f"{tax_year + 1}-02-28",
                "status": "required"
            },
            {
                "form": "Schedule K-1",
                "description": "Partner's Share of Income (if partnership)",
                "due_date": f"{tax_year + 1}-03-15",
                "status": "optional"
            }
        ]
        
        return {
            "syndicate_id": syndicate_id,
            "tax_year": tax_year,
            "summary": summary,
            "distributions": distributions_summary,
            "member_1099s": member_1099s[:5] if include_member_details else [],  # Limit for privacy
            "total_members_reported": len(member_1099s),
            "tax_obligations": tax_obligations,
            "filing_requirements": filing_requirements
        }
        
    except Exception as e:
        logger.error(f"Error generating tax report: {str(e)}")
        return {"error": str(e)}


# ===== MCP Tool Registration =====

SYNDICATE_TOOLS = {
    "syndicate_create": {
        "handler": syndicate_create,
        "description": "Create a new trading syndicate",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Syndicate name"},
                "description": {"type": "string", "description": "Syndicate description"},
                "initial_capital": {"type": "number", "description": "Initial capital amount"},
                "bankroll_rules": {"type": "object", "description": "Custom bankroll rules"},
                "distribution_model": {"type": "string", "description": "Profit distribution model"}
            },
            "required": ["name", "description", "initial_capital"]
        }
    },
    "syndicate_add_member": {
        "handler": syndicate_add_member,
        "description": "Add a member to a syndicate",
        "inputSchema": {
            "type": "object",
            "properties": {
                "syndicate_id": {"type": "string"},
                "name": {"type": "string"},
                "email": {"type": "string"},
                "role": {"type": "string", "enum": ["lead_investor", "senior_analyst", "junior_analyst", "contributing_member", "observer"]},
                "initial_contribution": {"type": "number"}
            },
            "required": ["syndicate_id", "name", "email", "role", "initial_contribution"]
        }
    },
    "syndicate_update_member": {
        "handler": syndicate_update_member,
        "description": "Update member details or status",
        "inputSchema": {
            "type": "object",
            "properties": {
                "syndicate_id": {"type": "string"},
                "member_id": {"type": "string"},
                "new_role": {"type": "string"},
                "contribution_adjustment": {"type": "number"},
                "is_active": {"type": "boolean"},
                "authorized_by": {"type": "string"}
            },
            "required": ["syndicate_id", "member_id", "authorized_by"]
        }
    },
    "syndicate_member_performance": {
        "handler": syndicate_member_performance,
        "description": "Get member performance metrics",
        "inputSchema": {
            "type": "object",
            "properties": {
                "syndicate_id": {"type": "string"},
                "member_id": {"type": "string"}
            },
            "required": ["syndicate_id", "member_id"]
        }
    },
    "syndicate_list_members": {
        "handler": syndicate_list_members,
        "description": "List all syndicate members",
        "inputSchema": {
            "type": "object",
            "properties": {
                "syndicate_id": {"type": "string"},
                "include_inactive": {"type": "boolean", "default": False}
            },
            "required": ["syndicate_id"]
        }
    },
    "syndicate_allocate_funds": {
        "handler": syndicate_allocate_funds,
        "description": "Allocate funds for a betting opportunity",
        "inputSchema": {
            "type": "object",
            "properties": {
                "syndicate_id": {"type": "string"},
                "sport": {"type": "string"},
                "event": {"type": "string"},
                "bet_type": {"type": "string"},
                "selection": {"type": "string"},
                "odds": {"type": "number"},
                "probability": {"type": "number"},
                "edge": {"type": "number"},
                "confidence": {"type": "number"},
                "strategy": {"type": "string", "default": "kelly_criterion"}
            },
            "required": ["syndicate_id", "sport", "event", "bet_type", "selection", "odds", "probability", "edge", "confidence"]
        }
    },
    "syndicate_get_exposure": {
        "handler": syndicate_get_exposure,
        "description": "Get current capital exposure",
        "inputSchema": {
            "type": "object",
            "properties": {
                "syndicate_id": {"type": "string"}
            },
            "required": ["syndicate_id"]
        }
    },
    "syndicate_distribute_profits": {
        "handler": syndicate_distribute_profits,
        "description": "Distribute profits to members",
        "inputSchema": {
            "type": "object",
            "properties": {
                "syndicate_id": {"type": "string"},
                "total_profit": {"type": "number"},
                "distribution_model": {"type": "string", "default": "hybrid"},
                "authorized_by": {"type": "string"}
            },
            "required": ["syndicate_id", "total_profit", "authorized_by"]
        }
    },
    "syndicate_process_withdrawal": {
        "handler": syndicate_process_withdrawal,
        "description": "Process member withdrawal request",
        "inputSchema": {
            "type": "object",
            "properties": {
                "syndicate_id": {"type": "string"},
                "member_id": {"type": "string"},
                "amount": {"type": "number"},
                "is_emergency": {"type": "boolean", "default": False}
            },
            "required": ["syndicate_id", "member_id", "amount"]
        }
    },
    "syndicate_get_balance": {
        "handler": syndicate_get_balance,
        "description": "Get syndicate balance and financial status",
        "inputSchema": {
            "type": "object",
            "properties": {
                "syndicate_id": {"type": "string"}
            },
            "required": ["syndicate_id"]
        }
    },
    "syndicate_create_vote": {
        "handler": syndicate_create_vote,
        "description": "Create a governance vote",
        "inputSchema": {
            "type": "object",
            "properties": {
                "syndicate_id": {"type": "string"},
                "proposal_type": {"type": "string"},
                "proposal_details": {"type": "object"},
                "proposed_by": {"type": "string"},
                "voting_period_hours": {"type": "integer", "default": 24}
            },
            "required": ["syndicate_id", "proposal_type", "proposal_details", "proposed_by"]
        }
    },
    "syndicate_cast_vote": {
        "handler": syndicate_cast_vote,
        "description": "Cast a vote on a proposal",
        "inputSchema": {
            "type": "object",
            "properties": {
                "syndicate_id": {"type": "string"},
                "vote_id": {"type": "string"},
                "member_id": {"type": "string"},
                "decision": {"type": "string", "enum": ["approve", "reject", "abstain"]}
            },
            "required": ["syndicate_id", "vote_id", "member_id", "decision"]
        }
    },
    "syndicate_vote_results": {
        "handler": syndicate_vote_results,
        "description": "Get voting results",
        "inputSchema": {
            "type": "object",
            "properties": {
                "syndicate_id": {"type": "string"},
                "vote_id": {"type": "string"}
            },
            "required": ["syndicate_id", "vote_id"]
        }
    },
    "syndicate_get_rules": {
        "handler": syndicate_get_rules,
        "description": "Get syndicate governance rules",
        "inputSchema": {
            "type": "object",
            "properties": {
                "syndicate_id": {"type": "string"}
            },
            "required": ["syndicate_id"]
        }
    },
    "syndicate_performance_report": {
        "handler": syndicate_performance_report,
        "description": "Generate comprehensive performance report",
        "inputSchema": {
            "type": "object",
            "properties": {
                "syndicate_id": {"type": "string"},
                "period_days": {"type": "integer", "default": 30}
            },
            "required": ["syndicate_id"]
        }
    },
    "syndicate_risk_analysis": {
        "handler": syndicate_risk_analysis,
        "description": "Perform risk analysis",
        "inputSchema": {
            "type": "object",
            "properties": {
                "syndicate_id": {"type": "string"},
                "include_scenarios": {"type": "boolean", "default": True}
            },
            "required": ["syndicate_id"]
        }
    },
    "syndicate_tax_report": {
        "handler": syndicate_tax_report,
        "description": "Generate tax report for syndicate",
        "inputSchema": {
            "type": "object",
            "properties": {
                "syndicate_id": {"type": "string"},
                "tax_year": {"type": "integer"},
                "include_member_details": {"type": "boolean", "default": True}
            },
            "required": ["syndicate_id"]
        }
    }
}


def register_syndicate_tools(server):
    """Register all syndicate tools with the MCP server"""
    for tool_name, tool_config in SYNDICATE_TOOLS.items():
        logger.info(f"Registering syndicate tool: {tool_name}")
        # Register with server's tool handler if available
        if hasattr(server, 'tools_handler'):
            server.tools_handler.register_tool(tool_name, tool_config)