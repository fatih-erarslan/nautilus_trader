"""
MCP Tools for Syndicate Investment System
Exposes syndicate functionality as callable tools for the MCP server
"""

from typing import Dict, Any, List, Optional
from decimal import Decimal
from datetime import datetime, timedelta
import json
import logging

from .member_management import (
    SyndicateMemberManager, 
    MemberRole, 
    MemberTier
)
from .capital_management import (
    FundAllocationEngine,
    ProfitDistributionSystem,
    WithdrawalManager,
    BettingOpportunity,
    AllocationStrategy,
    DistributionModel
)

logger = logging.getLogger(__name__)

# Global syndicate managers (in production, these would be persisted)
SYNDICATE_MANAGERS = {}
ALLOCATION_ENGINES = {}
DISTRIBUTION_SYSTEMS = {}
WITHDRAWAL_MANAGERS = {}


def create_syndicate(syndicate_id: str, name: str, description: str = "") -> Dict[str, Any]:
    """Create a new investment syndicate."""
    try:
        if syndicate_id in SYNDICATE_MANAGERS:
            return {"error": f"Syndicate {syndicate_id} already exists", "status": "failed"}
        
        manager = SyndicateMemberManager(syndicate_id)
        SYNDICATE_MANAGERS[syndicate_id] = manager
        
        return {
            "syndicate_id": syndicate_id,
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "total_members": 0,
            "total_capital": 0
        }
    except Exception as e:
        logger.error(f"Error creating syndicate: {e}")
        return {"error": str(e), "status": "failed"}


def add_member(syndicate_id: str, name: str, email: str, role: str, 
               initial_contribution: float) -> Dict[str, Any]:
    """Add a new member to a syndicate."""
    try:
        if syndicate_id not in SYNDICATE_MANAGERS:
            return {"error": f"Syndicate {syndicate_id} not found", "status": "failed"}
        
        manager = SYNDICATE_MANAGERS[syndicate_id]
        member_role = MemberRole(role)
        
        member = manager.add_member(
            name=name,
            email=email,
            role=member_role,
            initial_contribution=Decimal(str(initial_contribution))
        )
        
        return {
            "member_id": member.member_id,
            "name": member.name,
            "email": member.email,
            "role": member.role.value,
            "tier": member.tier.value,
            "capital_contribution": float(member.capital_contribution),
            "joined_at": member.joined_at.isoformat(),
            "status": "active"
        }
    except Exception as e:
        logger.error(f"Error adding member: {e}")
        return {"error": str(e), "status": "failed"}


def get_syndicate_status(syndicate_id: str) -> Dict[str, Any]:
    """Get current status and statistics for a syndicate."""
    try:
        if syndicate_id not in SYNDICATE_MANAGERS:
            return {"error": f"Syndicate {syndicate_id} not found", "status": "failed"}
        
        manager = SYNDICATE_MANAGERS[syndicate_id]
        stats = manager.get_syndicate_stats()
        
        return {
            "syndicate_id": syndicate_id,
            "total_members": stats["total_members"],
            "total_capital": float(stats["total_capital"]),
            "members_by_role": stats["members_by_role"],
            "members_by_tier": stats["members_by_tier"],
            "average_contribution": float(stats["average_contribution"]),
            "status": "active"
        }
    except Exception as e:
        logger.error(f"Error getting syndicate status: {e}")
        return {"error": str(e), "status": "failed"}


def allocate_funds(syndicate_id: str, opportunities: List[Dict[str, Any]], 
                  strategy: str = "kelly_criterion") -> Dict[str, Any]:
    """Allocate syndicate funds across betting opportunities."""
    try:
        if syndicate_id not in SYNDICATE_MANAGERS:
            return {"error": f"Syndicate {syndicate_id} not found", "status": "failed"}
        
        manager = SYNDICATE_MANAGERS[syndicate_id]
        total_capital = manager.get_total_capital()
        
        # Initialize allocation engine if needed
        if syndicate_id not in ALLOCATION_ENGINES:
            ALLOCATION_ENGINES[syndicate_id] = FundAllocationEngine(
                syndicate_id=syndicate_id,
                total_bankroll=total_capital
            )
        
        allocator = ALLOCATION_ENGINES[syndicate_id]
        allocator.update_bankroll(total_capital)
        
        # Convert opportunities to BettingOpportunity objects
        betting_opps = []
        for opp in opportunities:
            betting_opps.append(BettingOpportunity(
                sport=opp["sport"],
                event=opp["event"],
                bet_type=opp["bet_type"],
                selection=opp["selection"],
                odds=opp["odds"],
                probability=opp["probability"],
                edge=opp.get("edge", 0.0),
                confidence=opp.get("confidence", 0.5),
                model_agreement=opp.get("model_agreement", 0.5),
                time_until_event=timedelta(hours=opp.get("hours_until_event", 24)),
                liquidity=opp.get("liquidity", 10000)
            ))
        
        # Allocate funds
        allocation_strategy = AllocationStrategy(strategy)
        allocations = allocator.allocate_funds(betting_opps, allocation_strategy)
        
        # Format results
        results = []
        for alloc in allocations:
            results.append({
                "opportunity": {
                    "sport": alloc.opportunity.sport,
                    "event": alloc.opportunity.event,
                    "selection": alloc.opportunity.selection,
                    "odds": alloc.opportunity.odds,
                    "probability": alloc.opportunity.probability,
                    "edge": alloc.opportunity.edge
                },
                "allocated_amount": float(alloc.allocated_amount),
                "percentage_of_bankroll": float(alloc.percentage_of_bankroll),
                "expected_value": float(alloc.expected_value),
                "confidence_score": float(alloc.confidence_score),
                "risk_adjusted": alloc.risk_adjusted,
                "notes": alloc.notes
            })
        
        return {
            "syndicate_id": syndicate_id,
            "strategy": strategy,
            "total_capital": float(total_capital),
            "total_allocated": sum(float(a.allocated_amount) for a in allocations),
            "allocations": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error allocating funds: {e}")
        return {"error": str(e), "status": "failed"}


def distribute_profits(syndicate_id: str, total_profit: float, 
                      model: str = "hybrid") -> Dict[str, Any]:
    """Distribute profits among syndicate members."""
    try:
        if syndicate_id not in SYNDICATE_MANAGERS:
            return {"error": f"Syndicate {syndicate_id} not found", "status": "failed"}
        
        manager = SYNDICATE_MANAGERS[syndicate_id]
        
        # Initialize distribution system if needed
        if syndicate_id not in DISTRIBUTION_SYSTEMS:
            DISTRIBUTION_SYSTEMS[syndicate_id] = ProfitDistributionSystem(
                syndicate_id=syndicate_id,
                member_manager=manager
            )
        
        distributor = DISTRIBUTION_SYSTEMS[syndicate_id]
        distribution_model = DistributionModel(model)
        
        # Calculate distributions
        distributions = distributor.calculate_distribution(
            total_profit=Decimal(str(total_profit)),
            model=distribution_model
        )
        
        # Execute distribution
        success = distributor.execute_distribution(distributions)
        
        # Format results
        member_distributions = []
        for member_id, amount in distributions.items():
            member = manager.get_member(member_id)
            if member:
                member_distributions.append({
                    "member_id": member_id,
                    "member_name": member.name,
                    "role": member.role.value,
                    "tier": member.tier.value,
                    "gross_amount": float(amount),
                    "net_amount": float(amount * Decimal("0.9")),  # Simplified tax
                    "tax_withheld": float(amount * Decimal("0.1"))
                })
        
        return {
            "syndicate_id": syndicate_id,
            "total_profit": total_profit,
            "distribution_model": model,
            "distributions": member_distributions,
            "total_distributed": sum(d["gross_amount"] for d in member_distributions),
            "execution_status": "completed" if success else "failed",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error distributing profits: {e}")
        return {"error": str(e), "status": "failed"}


def process_withdrawal(syndicate_id: str, member_id: str, 
                      amount: float, is_emergency: bool = False) -> Dict[str, Any]:
    """Process a member withdrawal request."""
    try:
        if syndicate_id not in SYNDICATE_MANAGERS:
            return {"error": f"Syndicate {syndicate_id} not found", "status": "failed"}
        
        manager = SYNDICATE_MANAGERS[syndicate_id]
        member = manager.get_member(member_id)
        if not member:
            return {"error": f"Member {member_id} not found", "status": "failed"}
        
        # Initialize withdrawal manager if needed
        if syndicate_id not in WITHDRAWAL_MANAGERS:
            WITHDRAWAL_MANAGERS[syndicate_id] = WithdrawalManager(
                syndicate_id=syndicate_id,
                member_manager=manager
            )
        
        withdrawal_mgr = WITHDRAWAL_MANAGERS[syndicate_id]
        
        # Process withdrawal
        request = withdrawal_mgr.request_withdrawal(
            member_id=member_id,
            amount=Decimal(str(amount)),
            is_emergency=is_emergency
        )
        
        if request["status"] == "approved":
            # Execute withdrawal
            execution = withdrawal_mgr.execute_withdrawal(request["request_id"])
            
            return {
                "request_id": request["request_id"],
                "member_id": member_id,
                "member_name": member.name,
                "requested_amount": amount,
                "approved_amount": float(request["approved_amount"]),
                "penalty": float(request.get("penalty", 0)),
                "net_amount": float(request["approved_amount"] - request.get("penalty", 0)),
                "status": execution["status"],
                "execution_time": execution.get("execution_time", ""),
                "notes": request.get("notes", "")
            }
        else:
            return {
                "request_id": request.get("request_id", ""),
                "member_id": member_id,
                "requested_amount": amount,
                "status": request["status"],
                "reason": request.get("reason", "Unknown error"),
                "notes": request.get("notes", "")
            }
    except Exception as e:
        logger.error(f"Error processing withdrawal: {e}")
        return {"error": str(e), "status": "failed"}


def get_member_performance(syndicate_id: str, member_id: str) -> Dict[str, Any]:
    """Get detailed performance metrics for a syndicate member."""
    try:
        if syndicate_id not in SYNDICATE_MANAGERS:
            return {"error": f"Syndicate {syndicate_id} not found", "status": "failed"}
        
        manager = SYNDICATE_MANAGERS[syndicate_id]
        member = manager.get_member(member_id)
        if not member:
            return {"error": f"Member {member_id} not found", "status": "failed"}
        
        performance = member.performance_metrics
        
        return {
            "member_id": member_id,
            "name": member.name,
            "role": member.role.value,
            "tier": member.tier.value,
            "performance_metrics": {
                "total_profit": float(performance.total_profit),
                "roi": float(performance.roi),
                "win_rate": float(performance.win_rate),
                "average_return": float(performance.average_return),
                "sharpe_ratio": float(performance.sharpe_ratio),
                "alpha": float(performance.alpha),
                "skill_score": float(performance.skill_score),
                "consistency_score": float(performance.consistency_score),
                "risk_score": float(performance.risk_score)
            },
            "capital_contribution": float(member.capital_contribution),
            "lifetime_earnings": float(member.lifetime_earnings),
            "status": member.status.value,
            "joined_at": member.joined_at.isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting member performance: {e}")
        return {"error": str(e), "status": "failed"}


def create_vote(syndicate_id: str, vote_type: str, proposal: str, 
               options: List[str], duration_hours: int = 48) -> Dict[str, Any]:
    """Create a new vote for syndicate members."""
    # Voting functionality not yet implemented in member_management
    return {
        "error": "Voting functionality not yet implemented",
        "status": "failed",
        "message": "This feature will be available in a future update"
    }


def cast_vote(syndicate_id: str, vote_id: str, member_id: str, 
             option: str) -> Dict[str, Any]:
    """Cast a vote on a syndicate proposal."""
    # Voting functionality not yet implemented in member_management
    return {
        "error": "Voting functionality not yet implemented",
        "status": "failed",
        "message": "This feature will be available in a future update"
    }


def get_allocation_limits(syndicate_id: str) -> Dict[str, Any]:
    """Get current allocation limits and constraints for the syndicate."""
    try:
        if syndicate_id not in ALLOCATION_ENGINES:
            if syndicate_id not in SYNDICATE_MANAGERS:
                return {"error": f"Syndicate {syndicate_id} not found", "status": "failed"}
            
            manager = SYNDICATE_MANAGERS[syndicate_id]
            ALLOCATION_ENGINES[syndicate_id] = FundAllocationEngine(
                syndicate_id=syndicate_id,
                total_bankroll=manager.get_total_capital()
            )
        
        allocator = ALLOCATION_ENGINES[syndicate_id]
        
        return {
            "syndicate_id": syndicate_id,
            "limits": {
                "max_single_bet_percentage": allocator.max_single_bet_percentage,
                "max_parlay_percentage": allocator.max_parlay_percentage,
                "max_daily_exposure": allocator.max_daily_exposure,
                "max_sport_concentration": allocator.max_sport_concentration,
                "min_cash_reserve": allocator.min_cash_reserve,
                "daily_stop_loss": allocator.daily_stop_loss,
                "weekly_stop_loss": allocator.weekly_stop_loss
            },
            "current_exposure": {
                "daily": float(allocator.current_daily_exposure),
                "by_sport": {k: float(v) for k, v in allocator.exposure_by_sport.items()},
                "by_market": {(k.value if hasattr(k, 'value') else str(k)): float(v) for k, v in allocator.exposure_by_market.items()}
            },
            "available_capital": float(allocator.total_bankroll - allocator.current_daily_exposure),
            "cash_reserve": float(allocator.total_bankroll * allocator.min_cash_reserve)
        }
    except Exception as e:
        logger.error(f"Error getting allocation limits: {e}")
        return {"error": str(e), "status": "failed"}


def update_member_contribution(syndicate_id: str, member_id: str, 
                             additional_amount: float) -> Dict[str, Any]:
    """Update a member's capital contribution."""
    try:
        if syndicate_id not in SYNDICATE_MANAGERS:
            return {"error": f"Syndicate {syndicate_id} not found", "status": "failed"}
        
        manager = SYNDICATE_MANAGERS[syndicate_id]
        
        success = manager.update_member_contribution(
            member_id=member_id,
            additional_amount=Decimal(str(additional_amount))
        )
        
        if success:
            member = manager.get_member(member_id)
            return {
                "member_id": member_id,
                "name": member.name,
                "previous_contribution": float(member.capital_contribution - Decimal(str(additional_amount))),
                "additional_amount": additional_amount,
                "new_total_contribution": float(member.capital_contribution),
                "new_tier": member.tier.value,
                "status": "updated"
            }
        
        return {"error": "Failed to update contribution", "status": "failed"}
    except Exception as e:
        logger.error(f"Error updating member contribution: {e}")
        return {"error": str(e), "status": "failed"}


def get_profit_history(syndicate_id: str, days: int = 30) -> Dict[str, Any]:
    """Get profit distribution history for the syndicate."""
    try:
        if syndicate_id not in DISTRIBUTION_SYSTEMS:
            if syndicate_id not in SYNDICATE_MANAGERS:
                return {"error": f"Syndicate {syndicate_id} not found", "status": "failed"}
            
            manager = SYNDICATE_MANAGERS[syndicate_id]
            DISTRIBUTION_SYSTEMS[syndicate_id] = ProfitDistributionSystem(
                syndicate_id=syndicate_id,
                member_manager=manager
            )
        
        distributor = DISTRIBUTION_SYSTEMS[syndicate_id]
        
        # Get recent distributions
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_distributions = [
            d for d in distributor.distribution_history 
            if d["timestamp"] >= cutoff_date
        ]
        
        # Calculate summary statistics
        total_distributed = sum(d["total_amount"] for d in recent_distributions)
        distribution_count = len(recent_distributions)
        
        return {
            "syndicate_id": syndicate_id,
            "period_days": days,
            "summary": {
                "total_distributed": float(total_distributed),
                "distribution_count": distribution_count,
                "average_distribution": float(total_distributed / distribution_count) if distribution_count > 0 else 0
            },
            "distributions": [
                {
                    "timestamp": d["timestamp"].isoformat(),
                    "total_amount": float(d["total_amount"]),
                    "model": d["model"],
                    "member_count": d["member_count"]
                }
                for d in recent_distributions
            ]
        }
    except Exception as e:
        logger.error(f"Error getting profit history: {e}")
        return {"error": str(e), "status": "failed"}


def simulate_allocation(syndicate_id: str, opportunities: List[Dict[str, Any]], 
                       test_strategies: List[str] = None) -> Dict[str, Any]:
    """Simulate fund allocation across multiple strategies."""
    try:
        if syndicate_id not in SYNDICATE_MANAGERS:
            return {"error": f"Syndicate {syndicate_id} not found", "status": "failed"}
        
        manager = SYNDICATE_MANAGERS[syndicate_id]
        total_capital = manager.get_total_capital()
        
        # Default strategies if none provided
        if not test_strategies:
            test_strategies = ["kelly_criterion", "fixed_percentage", "dynamic_confidence"]
        
        # Convert opportunities
        betting_opps = []
        for opp in opportunities:
            betting_opps.append(BettingOpportunity(
                sport=opp["sport"],
                event=opp["event"],
                bet_type=opp["bet_type"],
                selection=opp["selection"],
                odds=opp["odds"],
                probability=opp["probability"],
                edge=opp.get("edge", 0.0),
                confidence=opp.get("confidence", 0.5),
                model_agreement=opp.get("model_agreement", 0.5),
                time_until_event=timedelta(hours=opp.get("hours_until_event", 24)),
                liquidity=opp.get("liquidity", 10000)
            ))
        
        # Simulate each strategy
        simulations = {}
        for strategy_name in test_strategies:
            try:
                # Create temporary allocator
                temp_allocator = FundAllocationEngine(
                    syndicate_id=f"{syndicate_id}_sim",
                    total_bankroll=total_capital
                )
                
                strategy = AllocationStrategy(strategy_name)
                allocations = temp_allocator.allocate_funds(betting_opps, strategy)
                
                total_allocated = sum(a.allocated_amount for a in allocations)
                total_ev = sum(a.expected_value for a in allocations)
                
                simulations[strategy_name] = {
                    "total_allocated": float(total_allocated),
                    "allocation_percentage": float(total_allocated / total_capital * 100),
                    "expected_value": float(total_ev),
                    "expected_roi": float(total_ev / total_allocated * 100) if total_allocated > 0 else 0,
                    "bet_count": len([a for a in allocations if a.allocated_amount > 0]),
                    "allocations": [
                        {
                            "event": a.opportunity.event,
                            "selection": a.opportunity.selection,
                            "amount": float(a.allocated_amount),
                            "ev": float(a.expected_value)
                        }
                        for a in allocations if a.allocated_amount > 0
                    ]
                }
            except Exception as e:
                simulations[strategy_name] = {"error": str(e)}
        
        return {
            "syndicate_id": syndicate_id,
            "total_capital": float(total_capital),
            "opportunity_count": len(opportunities),
            "simulations": simulations,
            "recommendation": max(simulations.items(), 
                                key=lambda x: x[1].get("expected_roi", 0) if "error" not in x[1] else -999)[0]
        }
    except Exception as e:
        logger.error(f"Error simulating allocation: {e}")
        return {"error": str(e), "status": "failed"}


def get_withdrawal_history(syndicate_id: str, member_id: Optional[str] = None) -> Dict[str, Any]:
    """Get withdrawal history for the syndicate or specific member."""
    try:
        if syndicate_id not in WITHDRAWAL_MANAGERS:
            if syndicate_id not in SYNDICATE_MANAGERS:
                return {"error": f"Syndicate {syndicate_id} not found", "status": "failed"}
            
            manager = SYNDICATE_MANAGERS[syndicate_id]
            WITHDRAWAL_MANAGERS[syndicate_id] = WithdrawalManager(
                syndicate_id=syndicate_id,
                member_manager=manager
            )
        
        withdrawal_mgr = WITHDRAWAL_MANAGERS[syndicate_id]
        
        # Filter by member if specified
        if member_id:
            requests = [r for r in withdrawal_mgr.withdrawal_requests.values() 
                       if r["member_id"] == member_id]
        else:
            requests = list(withdrawal_mgr.withdrawal_requests.values())
        
        # Format results
        formatted_requests = []
        for req in requests:
            formatted_requests.append({
                "request_id": req["request_id"],
                "member_id": req["member_id"],
                "requested_amount": float(req["requested_amount"]),
                "approved_amount": float(req.get("approved_amount", 0)),
                "penalty": float(req.get("penalty", 0)),
                "status": req["status"],
                "is_emergency": req.get("is_emergency", False),
                "requested_at": req["requested_at"].isoformat(),
                "processed_at": req.get("processed_at", "").isoformat() if req.get("processed_at") else None,
                "notes": req.get("notes", "")
            })
        
        # Calculate summary
        total_requested = sum(r["requested_amount"] for r in formatted_requests)
        total_approved = sum(r["approved_amount"] for r in formatted_requests 
                           if r["status"] == "completed")
        total_penalties = sum(r["penalty"] for r in formatted_requests)
        
        return {
            "syndicate_id": syndicate_id,
            "member_id": member_id,
            "summary": {
                "total_requests": len(formatted_requests),
                "total_requested": total_requested,
                "total_approved": total_approved,
                "total_penalties": total_penalties,
                "pending_requests": len([r for r in formatted_requests if r["status"] == "pending"])
            },
            "requests": formatted_requests
        }
    except Exception as e:
        logger.error(f"Error getting withdrawal history: {e}")
        return {"error": str(e), "status": "failed"}


def update_allocation_strategy(syndicate_id: str, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
    """Update allocation strategy parameters for the syndicate."""
    try:
        if syndicate_id not in ALLOCATION_ENGINES:
            if syndicate_id not in SYNDICATE_MANAGERS:
                return {"error": f"Syndicate {syndicate_id} not found", "status": "failed"}
            
            manager = SYNDICATE_MANAGERS[syndicate_id]
            ALLOCATION_ENGINES[syndicate_id] = FundAllocationEngine(
                syndicate_id=syndicate_id,
                total_bankroll=manager.get_total_capital()
            )
        
        allocator = ALLOCATION_ENGINES[syndicate_id]
        
        # Update configurable parameters
        updates = {}
        if "max_single_bet_percentage" in strategy_config:
            allocator.max_single_bet_percentage = strategy_config["max_single_bet_percentage"]
            updates["max_single_bet_percentage"] = allocator.max_single_bet_percentage
        
        if "max_daily_exposure" in strategy_config:
            allocator.max_daily_exposure = strategy_config["max_daily_exposure"]
            updates["max_daily_exposure"] = allocator.max_daily_exposure
        
        if "min_cash_reserve" in strategy_config:
            allocator.min_cash_reserve = strategy_config["min_cash_reserve"]
            updates["min_cash_reserve"] = allocator.min_cash_reserve
        
        if "daily_stop_loss" in strategy_config:
            allocator.daily_stop_loss = strategy_config["daily_stop_loss"]
            updates["daily_stop_loss"] = allocator.daily_stop_loss
        
        if "weekly_stop_loss" in strategy_config:
            allocator.weekly_stop_loss = strategy_config["weekly_stop_loss"]
            updates["weekly_stop_loss"] = allocator.weekly_stop_loss
        
        return {
            "syndicate_id": syndicate_id,
            "updates": updates,
            "status": "updated",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error updating allocation strategy: {e}")
        return {"error": str(e), "status": "failed"}


def get_member_list(syndicate_id: str, active_only: bool = True) -> Dict[str, Any]:
    """Get list of all members in the syndicate."""
    try:
        if syndicate_id not in SYNDICATE_MANAGERS:
            return {"error": f"Syndicate {syndicate_id} not found", "status": "failed"}
        
        manager = SYNDICATE_MANAGERS[syndicate_id]
        members = manager.get_all_members()
        
        if active_only:
            members = [m for m in members if m.status.value == "active"]
        
        member_list = []
        for member in members:
            member_list.append({
                "member_id": member.member_id,
                "name": member.name,
                "email": member.email,
                "role": member.role.value,
                "tier": member.tier.value,
                "capital_contribution": float(member.capital_contribution),
                "lifetime_earnings": float(member.lifetime_earnings),
                "roi": float(member.performance_metrics.roi),
                "status": member.status.value,
                "joined_at": member.joined_at.isoformat()
            })
        
        return {
            "syndicate_id": syndicate_id,
            "total_members": len(member_list),
            "active_members": len([m for m in member_list if m["status"] == "active"]),
            "members": member_list
        }
    except Exception as e:
        logger.error(f"Error getting member list: {e}")
        return {"error": str(e), "status": "failed"}


def calculate_tax_liability(syndicate_id: str, member_id: str, 
                          jurisdiction: str = "US") -> Dict[str, Any]:
    """Calculate estimated tax liability for a member's earnings."""
    try:
        if syndicate_id not in SYNDICATE_MANAGERS:
            return {"error": f"Syndicate {syndicate_id} not found", "status": "failed"}
        
        manager = SYNDICATE_MANAGERS[syndicate_id]
        member = manager.get_member(member_id)
        if not member:
            return {"error": f"Member {member_id} not found", "status": "failed"}
        
        # Simplified tax calculation (in production, this would be much more complex)
        tax_rates = {
            "US": {"federal": 0.22, "state": 0.05, "medicare": 0.0145},
            "UK": {"income": 0.20, "ni": 0.12},
            "CA": {"federal": 0.15, "provincial": 0.10},
            "AU": {"income": 0.19, "medicare": 0.02}
        }
        
        if jurisdiction not in tax_rates:
            jurisdiction = "US"
        
        earnings = float(member.lifetime_earnings)
        rates = tax_rates[jurisdiction]
        
        tax_breakdown = {}
        total_tax = 0
        
        for tax_type, rate in rates.items():
            tax_amount = earnings * rate
            tax_breakdown[tax_type] = {
                "rate": rate,
                "amount": round(tax_amount, 2)
            }
            total_tax += tax_amount
        
        return {
            "member_id": member_id,
            "member_name": member.name,
            "jurisdiction": jurisdiction,
            "gross_earnings": earnings,
            "tax_breakdown": tax_breakdown,
            "total_tax": round(total_tax, 2),
            "net_earnings": round(earnings - total_tax, 2),
            "effective_rate": round(total_tax / earnings * 100, 2) if earnings > 0 else 0,
            "disclaimer": "This is an estimate only. Consult a tax professional."
        }
    except Exception as e:
        logger.error(f"Error calculating tax liability: {e}")
        return {"error": str(e), "status": "failed"}