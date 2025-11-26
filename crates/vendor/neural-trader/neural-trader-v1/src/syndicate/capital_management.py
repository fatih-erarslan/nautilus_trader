"""
Syndicate Capital Management System
Handles fund allocation, bankroll management, and profit distribution
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from dataclasses import dataclass
import json
import math
from enum import Enum


class AllocationStrategy(Enum):
    """Fund allocation strategies"""
    KELLY_CRITERION = "kelly_criterion"
    FIXED_PERCENTAGE = "fixed_percentage"
    DYNAMIC_CONFIDENCE = "dynamic_confidence"
    RISK_PARITY = "risk_parity"
    MARTINGALE = "martingale"
    ANTI_MARTINGALE = "anti_martingale"


class DistributionModel(Enum):
    """Profit distribution models"""
    PROPORTIONAL = "proportional"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    TIERED = "tiered"
    HYBRID = "hybrid"


@dataclass
class BankrollRules:
    """Bankroll management rules"""
    max_single_bet: float = 0.05  # 5% of total bankroll
    max_daily_exposure: float = 0.20  # 20% of bankroll
    max_sport_concentration: float = 0.40  # 40% in one sport
    minimum_reserve: float = 0.30  # 30% cash reserve
    stop_loss_daily: float = 0.10  # 10% daily loss limit
    stop_loss_weekly: float = 0.20  # 20% weekly loss limit
    profit_lock: float = 0.50  # Lock 50% of profits
    max_parlay_percentage: float = 0.02  # 2% for parlays
    max_live_betting: float = 0.15  # 15% for live betting


@dataclass
class BettingOpportunity:
    """Betting opportunity details"""
    sport: str
    event: str
    bet_type: str
    selection: str
    odds: float
    probability: float
    edge: float
    confidence: float
    model_agreement: float
    time_until_event: timedelta
    liquidity: float
    is_live: bool = False
    is_parlay: bool = False


@dataclass
class AllocationResult:
    """Result of fund allocation"""
    amount: Decimal
    percentage_of_bankroll: float
    reasoning: Dict[str, Any]
    risk_metrics: Dict[str, float]
    approval_required: bool
    warnings: List[str]
    recommended_stake_sizing: Dict[str, Decimal]


class FundAllocationEngine:
    """Automated fund allocation system"""
    
    def __init__(self, syndicate_id: str, total_bankroll: Decimal):
        self.syndicate_id = syndicate_id
        self.total_bankroll = total_bankroll
        self.rules = BankrollRules()
        self.current_exposure = self._initialize_exposure_tracking()
        self.allocation_history = []
        
    def _initialize_exposure_tracking(self) -> Dict[str, Decimal]:
        """Initialize exposure tracking"""
        return {
            "daily": Decimal("0"),
            "weekly": Decimal("0"),
            "by_sport": {},
            "live_betting": Decimal("0"),
            "parlays": Decimal("0"),
            "open_bets": []
        }
    
    def allocate_funds(self, opportunity: BettingOpportunity, 
                      strategy: AllocationStrategy = AllocationStrategy.KELLY_CRITERION) -> AllocationResult:
        """Allocate funds for a betting opportunity"""
        
        # Calculate base allocation
        base_allocation = self._calculate_base_allocation(opportunity, strategy)
        
        # Apply constraints and safety checks
        constrained_allocation = self._apply_constraints(base_allocation, opportunity)
        
        # Generate reasoning and risk metrics
        reasoning = self._generate_allocation_reasoning(opportunity, strategy, base_allocation, constrained_allocation)
        risk_metrics = self._calculate_risk_metrics(constrained_allocation, opportunity)
        
        # Check if approval is required
        approval_required = self._needs_approval(constrained_allocation)
        
        # Generate warnings
        warnings = self._generate_warnings(constrained_allocation, opportunity)
        
        # Calculate recommended stake sizing options
        stake_sizing = self._calculate_stake_sizing_options(constrained_allocation, opportunity)
        
        result = AllocationResult(
            amount=constrained_allocation,
            percentage_of_bankroll=float(constrained_allocation / self.total_bankroll),
            reasoning=reasoning,
            risk_metrics=risk_metrics,
            approval_required=approval_required,
            warnings=warnings,
            recommended_stake_sizing=stake_sizing
        )
        
        # Log allocation
        self._log_allocation(result, opportunity)
        
        return result
    
    def _calculate_base_allocation(self, opportunity: BettingOpportunity, 
                                 strategy: AllocationStrategy) -> Decimal:
        """Calculate base allocation amount"""
        
        if strategy == AllocationStrategy.KELLY_CRITERION:
            return self._kelly_allocation(opportunity)
        elif strategy == AllocationStrategy.FIXED_PERCENTAGE:
            return self._fixed_allocation(opportunity)
        elif strategy == AllocationStrategy.DYNAMIC_CONFIDENCE:
            return self._confidence_based_allocation(opportunity)
        elif strategy == AllocationStrategy.RISK_PARITY:
            return self._risk_parity_allocation(opportunity)
        else:
            return self._kelly_allocation(opportunity)  # Default to Kelly
    
    def _kelly_allocation(self, opportunity: BettingOpportunity) -> Decimal:
        """Kelly Criterion allocation with fractional betting"""
        if opportunity.edge <= 0 or opportunity.probability <= 0:
            return Decimal("0")
        
        # Kelly percentage = (bp - q) / b
        # where b = decimal odds - 1, p = probability of winning, q = 1 - p
        b = opportunity.odds - 1
        p = opportunity.probability
        q = 1 - p
        
        kelly_percentage = (b * p - q) / b
        
        # Apply fractional Kelly (25% of full Kelly for safety)
        conservative_kelly = kelly_percentage * 0.25
        
        # Adjust for confidence and model agreement
        confidence_adjustment = opportunity.confidence * opportunity.model_agreement
        adjusted_kelly = conservative_kelly * confidence_adjustment
        
        # Convert to amount
        allocation = self.total_bankroll * Decimal(str(max(0, adjusted_kelly)))
        
        return allocation.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    
    def _fixed_allocation(self, opportunity: BettingOpportunity) -> Decimal:
        """Fixed percentage allocation"""
        base_percentage = Decimal("0.02")  # 2% base
        
        # Adjust for confidence
        confidence_multiplier = Decimal(str(opportunity.confidence))
        
        # Adjust for edge
        edge_multiplier = Decimal(str(1 + opportunity.edge))
        
        adjusted_percentage = base_percentage * confidence_multiplier * edge_multiplier
        
        allocation = self.total_bankroll * adjusted_percentage
        
        return allocation.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    
    def _confidence_based_allocation(self, opportunity: BettingOpportunity) -> Decimal:
        """Allocation based on confidence levels"""
        confidence_tiers = {
            0.9: Decimal("0.05"),  # Very high confidence: 5%
            0.8: Decimal("0.04"),  # High confidence: 4%
            0.7: Decimal("0.03"),  # Good confidence: 3%
            0.6: Decimal("0.02"),  # Moderate confidence: 2%
            0.5: Decimal("0.01"),  # Low confidence: 1%
        }
        
        # Find appropriate tier
        allocation_percentage = Decimal("0.005")  # Default 0.5%
        for conf_threshold, percentage in sorted(confidence_tiers.items(), reverse=True):
            if opportunity.confidence >= conf_threshold:
                allocation_percentage = percentage
                break
        
        # Adjust for edge
        if opportunity.edge > 0.1:  # Strong edge
            allocation_percentage *= Decimal("1.5")
        elif opportunity.edge > 0.05:  # Good edge
            allocation_percentage *= Decimal("1.25")
        
        allocation = self.total_bankroll * allocation_percentage
        
        return allocation.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    
    def _risk_parity_allocation(self, opportunity: BettingOpportunity) -> Decimal:
        """Risk parity allocation considering portfolio risk"""
        # Target risk contribution
        target_risk_contribution = Decimal("0.01")  # 1% risk contribution
        
        # Estimate bet volatility (simplified)
        bet_volatility = Decimal(str(1 / math.sqrt(opportunity.odds)))
        
        # Calculate allocation to achieve target risk
        allocation = (target_risk_contribution * self.total_bankroll) / bet_volatility
        
        # Adjust for correlation with existing bets
        correlation_adjustment = self._calculate_correlation_adjustment(opportunity)
        allocation *= correlation_adjustment
        
        return allocation.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    
    def _apply_constraints(self, base_allocation: Decimal, 
                         opportunity: BettingOpportunity) -> Decimal:
        """Apply bankroll management constraints"""
        allocation = base_allocation
        
        # Maximum single bet constraint
        max_single = self.total_bankroll * Decimal(str(self.rules.max_single_bet))
        if opportunity.is_parlay:
            max_single = self.total_bankroll * Decimal(str(self.rules.max_parlay_percentage))
        
        allocation = min(allocation, max_single)
        
        # Daily exposure constraint
        remaining_daily = (self.total_bankroll * Decimal(str(self.rules.max_daily_exposure))) - self.current_exposure["daily"]
        allocation = min(allocation, max(Decimal("0"), remaining_daily))
        
        # Sport concentration constraint
        sport_exposure = self.current_exposure["by_sport"].get(opportunity.sport, Decimal("0"))
        max_sport = self.total_bankroll * Decimal(str(self.rules.max_sport_concentration))
        remaining_sport = max_sport - sport_exposure
        allocation = min(allocation, max(Decimal("0"), remaining_sport))
        
        # Minimum reserve constraint
        available_funds = self.total_bankroll - self._calculate_total_exposure()
        max_available = available_funds - (self.total_bankroll * Decimal(str(self.rules.minimum_reserve)))
        allocation = min(allocation, max(Decimal("0"), max_available))
        
        # Live betting constraint
        if opportunity.is_live:
            remaining_live = (self.total_bankroll * Decimal(str(self.rules.max_live_betting))) - self.current_exposure["live_betting"]
            allocation = min(allocation, max(Decimal("0"), remaining_live))
        
        # Stop loss check
        if self._check_stop_loss():
            allocation = Decimal("0")
        
        return allocation.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    
    def _calculate_correlation_adjustment(self, opportunity: BettingOpportunity) -> Decimal:
        """Calculate correlation adjustment for risk parity"""
        # Simplified correlation check
        same_sport_bets = [bet for bet in self.current_exposure["open_bets"] 
                          if bet["sport"] == opportunity.sport]
        
        if not same_sport_bets:
            return Decimal("1.0")
        
        # Reduce allocation based on number of correlated bets
        correlation_factor = Decimal(str(1.0 / (1.0 + len(same_sport_bets) * 0.2)))
        
        return correlation_factor
    
    def _calculate_total_exposure(self) -> Decimal:
        """Calculate total current exposure"""
        total = Decimal("0")
        for bet in self.current_exposure["open_bets"]:
            total += bet["amount"]
        return total
    
    def _check_stop_loss(self) -> bool:
        """Check if stop loss has been triggered"""
        # This would check actual P&L
        # Placeholder implementation
        return False
    
    def _needs_approval(self, allocation: Decimal) -> bool:
        """Check if allocation needs approval"""
        # Large bets need approval
        if allocation > self.total_bankroll * Decimal("0.05"):
            return True
        
        # High daily exposure needs approval
        if (self.current_exposure["daily"] + allocation) > self.total_bankroll * Decimal("0.15"):
            return True
        
        return False
    
    def _generate_warnings(self, allocation: Decimal, opportunity: BettingOpportunity) -> List[str]:
        """Generate warnings for the allocation"""
        warnings = []
        
        # Check allocation size
        percentage = float(allocation / self.total_bankroll)
        if percentage > 0.04:
            warnings.append(f"Large bet size: {percentage:.1%} of bankroll")
        
        # Check daily exposure
        daily_percentage = float((self.current_exposure["daily"] + allocation) / self.total_bankroll)
        if daily_percentage > 0.15:
            warnings.append(f"High daily exposure: {daily_percentage:.1%}")
        
        # Check sport concentration
        sport_exposure = self.current_exposure["by_sport"].get(opportunity.sport, Decimal("0"))
        sport_percentage = float((sport_exposure + allocation) / self.total_bankroll)
        if sport_percentage > 0.30:
            warnings.append(f"High {opportunity.sport} concentration: {sport_percentage:.1%}")
        
        # Check edge
        if opportunity.edge < 0.03:
            warnings.append(f"Low edge: {opportunity.edge:.1%}")
        
        # Check liquidity
        if opportunity.liquidity < 10000:
            warnings.append(f"Low liquidity: ${opportunity.liquidity:,.0f}")
        
        return warnings
    
    def _calculate_risk_metrics(self, allocation: Decimal, opportunity: BettingOpportunity) -> Dict[str, float]:
        """Calculate risk metrics for the allocation"""
        metrics = {}
        
        # Expected value
        metrics["expected_value"] = float(allocation) * opportunity.edge
        
        # Value at Risk (simplified)
        metrics["value_at_risk_95"] = float(allocation) * 0.95  # 95% VaR
        
        # Probability of ruin (simplified Kelly)
        if opportunity.edge > 0:
            kelly_fraction = float(allocation / self.total_bankroll) / (opportunity.edge / (opportunity.odds - 1))
            metrics["kelly_fraction"] = kelly_fraction
            metrics["probability_of_ruin"] = math.exp(-2 * opportunity.edge * float(self.total_bankroll / allocation)) if allocation > 0 else 0
        else:
            metrics["kelly_fraction"] = 0
            metrics["probability_of_ruin"] = 1
        
        # Sharpe ratio estimate
        expected_return = opportunity.edge
        volatility = 1 / math.sqrt(opportunity.odds)
        metrics["estimated_sharpe"] = expected_return / volatility if volatility > 0 else 0
        
        # Risk-reward ratio
        potential_profit = float(allocation) * (opportunity.odds - 1)
        potential_loss = float(allocation)
        metrics["risk_reward_ratio"] = potential_profit / potential_loss
        
        return metrics
    
    def _generate_allocation_reasoning(self, opportunity: BettingOpportunity, 
                                     strategy: AllocationStrategy,
                                     base_allocation: Decimal, 
                                     final_allocation: Decimal) -> Dict[str, Any]:
        """Generate detailed reasoning for allocation"""
        reasoning = {
            "strategy_used": strategy.value,
            "base_calculation": {
                "method": strategy.value,
                "initial_amount": str(base_allocation),
                "edge": opportunity.edge,
                "confidence": opportunity.confidence,
                "model_agreement": opportunity.model_agreement
            },
            "constraints_applied": [],
            "final_amount": str(final_allocation),
            "reduction_percentage": float((base_allocation - final_allocation) / base_allocation * 100) if base_allocation > 0 else 0
        }
        
        # Document constraint applications
        if final_allocation < base_allocation:
            if final_allocation == self.total_bankroll * Decimal(str(self.rules.max_single_bet)):
                reasoning["constraints_applied"].append("Maximum single bet limit")
            
            if self.current_exposure["daily"] + final_allocation >= self.total_bankroll * Decimal(str(self.rules.max_daily_exposure)):
                reasoning["constraints_applied"].append("Daily exposure limit")
            
            sport_exposure = self.current_exposure["by_sport"].get(opportunity.sport, Decimal("0"))
            if sport_exposure + final_allocation >= self.total_bankroll * Decimal(str(self.rules.max_sport_concentration)):
                reasoning["constraints_applied"].append(f"{opportunity.sport} concentration limit")
        
        return reasoning
    
    def _calculate_stake_sizing_options(self, base_allocation: Decimal, 
                                      opportunity: BettingOpportunity) -> Dict[str, Decimal]:
        """Calculate different stake sizing options"""
        options = {
            "recommended": base_allocation,
            "conservative": base_allocation * Decimal("0.5"),
            "aggressive": min(base_allocation * Decimal("1.5"), 
                            self.total_bankroll * Decimal(str(self.rules.max_single_bet))),
            "minimum": min(base_allocation * Decimal("0.25"), Decimal("10"))  # $10 minimum
        }
        
        # Round all options
        for key in options:
            options[key] = options[key].quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        
        return options
    
    def update_exposure(self, bet_placed: Dict[str, Any]):
        """Update exposure tracking after bet placement"""
        amount = Decimal(str(bet_placed["amount"]))
        
        self.current_exposure["daily"] += amount
        self.current_exposure["weekly"] += amount
        
        sport = bet_placed["sport"]
        if sport not in self.current_exposure["by_sport"]:
            self.current_exposure["by_sport"][sport] = Decimal("0")
        self.current_exposure["by_sport"][sport] += amount
        
        if bet_placed.get("is_live", False):
            self.current_exposure["live_betting"] += amount
        
        if bet_placed.get("is_parlay", False):
            self.current_exposure["parlays"] += amount
        
        self.current_exposure["open_bets"].append({
            "bet_id": bet_placed["bet_id"],
            "sport": sport,
            "amount": amount,
            "placed_at": datetime.now()
        })
    
    def _log_allocation(self, result: AllocationResult, opportunity: BettingOpportunity):
        """Log allocation decision"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "syndicate_id": self.syndicate_id,
            "opportunity": {
                "sport": opportunity.sport,
                "event": opportunity.event,
                "odds": opportunity.odds,
                "edge": opportunity.edge
            },
            "allocation": {
                "amount": str(result.amount),
                "percentage": result.percentage_of_bankroll,
                "approval_required": result.approval_required
            },
            "warnings": result.warnings
        }
        
        self.allocation_history.append(log_entry)


class ProfitDistributionSystem:
    """Handle profit distribution among syndicate members"""
    
    def __init__(self, syndicate_id: str):
        self.syndicate_id = syndicate_id
        self.distribution_history = []
        self.tax_withholding_rates = {
            "US": 0.24,  # Federal rate
            "UK": 0.20,  # Basic rate
            "AU": 0.32,  # Average rate
            "CA": 0.25   # Average rate
        }
        
    def calculate_distribution(self, total_profit: Decimal, 
                             members: List[Dict[str, Any]],
                             model: DistributionModel = DistributionModel.HYBRID) -> Dict[str, Dict[str, Any]]:
        """Calculate profit distribution for all members"""
        
        # Reserve operational costs
        operational_reserve = total_profit * Decimal("0.05")  # 5% for operations
        distributable_profit = total_profit - operational_reserve
        
        # Calculate distributions based on model
        if model == DistributionModel.HYBRID:
            distributions = self._hybrid_distribution(distributable_profit, members)
        elif model == DistributionModel.PROPORTIONAL:
            distributions = self._proportional_distribution(distributable_profit, members)
        elif model == DistributionModel.PERFORMANCE_WEIGHTED:
            distributions = self._performance_weighted_distribution(distributable_profit, members)
        elif model == DistributionModel.TIERED:
            distributions = self._tiered_distribution(distributable_profit, members)
        else:
            distributions = self._hybrid_distribution(distributable_profit, members)
        
        # Apply tax withholding and finalize
        final_distributions = {}
        for member_id, gross_amount in distributions.items():
            member = next(m for m in members if m["id"] == member_id)
            tax_withheld = self._calculate_tax_withholding(gross_amount, member)
            
            final_distributions[member_id] = {
                "gross_amount": str(gross_amount),
                "tax_withheld": str(tax_withheld),
                "net_amount": str(gross_amount - tax_withheld),
                "payment_method": member.get("payment_method", "bank_transfer"),
                "payment_details": self._get_payment_details(member)
            }
        
        # Log distribution
        self._log_distribution(total_profit, operational_reserve, final_distributions, model)
        
        return final_distributions
    
    def _hybrid_distribution(self, profit: Decimal, members: List[Dict[str, Any]]) -> Dict[str, Decimal]:
        """Hybrid distribution: 50% capital, 30% performance, 20% equal"""
        distributions = {}
        
        total_capital = sum(Decimal(str(m["capital_contribution"])) for m in members if m["is_active"])
        total_performance_score = sum(m["performance_score"] for m in members if m["is_active"])
        active_members = [m for m in members if m["is_active"]]
        
        capital_portion = profit * Decimal("0.50")
        performance_portion = profit * Decimal("0.30")
        equal_portion = profit * Decimal("0.20")
        
        for member in active_members:
            member_capital = Decimal(str(member["capital_contribution"]))
            
            # Capital-based share
            capital_share = (member_capital / total_capital * capital_portion) if total_capital > 0 else Decimal("0")
            
            # Performance-based share
            if total_performance_score > 0:
                performance_share = (Decimal(str(member["performance_score"])) / Decimal(str(total_performance_score))) * performance_portion
            else:
                performance_share = performance_portion / len(active_members)
            
            # Equal share
            equal_share = equal_portion / len(active_members)
            
            # Total distribution
            total_share = capital_share + performance_share + equal_share
            distributions[member["id"]] = total_share.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        
        return distributions
    
    def _proportional_distribution(self, profit: Decimal, members: List[Dict[str, Any]]) -> Dict[str, Decimal]:
        """Pure proportional distribution based on capital contribution"""
        distributions = {}
        
        total_capital = sum(Decimal(str(m["capital_contribution"])) for m in members if m["is_active"])
        
        for member in members:
            if member["is_active"] and total_capital > 0:
                member_capital = Decimal(str(member["capital_contribution"]))
                share = (member_capital / total_capital) * profit
                distributions[member["id"]] = share.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        
        return distributions
    
    def _performance_weighted_distribution(self, profit: Decimal, 
                                         members: List[Dict[str, Any]]) -> Dict[str, Decimal]:
        """Distribution weighted heavily by performance"""
        distributions = {}
        
        # Calculate composite scores
        for member in members:
            if member["is_active"]:
                # Composite score: 60% ROI, 30% win rate, 10% consistency
                roi_score = member.get("roi_score", 0) * 0.6
                win_rate_score = member.get("win_rate", 0) * 0.3
                consistency_score = member.get("consistency_score", 0.5) * 0.1
                
                composite_score = roi_score + win_rate_score + consistency_score
                member["composite_score"] = composite_score
        
        total_score = sum(m["composite_score"] for m in members if m["is_active"])
        
        for member in members:
            if member["is_active"] and total_score > 0:
                share = (Decimal(str(member["composite_score"])) / Decimal(str(total_score))) * profit
                distributions[member["id"]] = share.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        
        return distributions
    
    def _tiered_distribution(self, profit: Decimal, members: List[Dict[str, Any]]) -> Dict[str, Decimal]:
        """Distribution based on member tiers"""
        distributions = {}
        
        # Tier weights
        tier_weights = {
            "platinum": 1.5,
            "gold": 1.2,
            "silver": 1.0,
            "bronze": 0.8
        }
        
        # Calculate weighted shares
        weighted_units = sum(tier_weights.get(m["tier"], 1.0) for m in members if m["is_active"])
        
        for member in members:
            if member["is_active"]:
                weight = tier_weights.get(member["tier"], 1.0)
                share = (Decimal(str(weight)) / Decimal(str(weighted_units))) * profit
                distributions[member["id"]] = share.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        
        return distributions
    
    def _calculate_tax_withholding(self, amount: Decimal, member: Dict[str, Any]) -> Decimal:
        """Calculate tax withholding amount"""
        jurisdiction = member.get("tax_jurisdiction", "US")
        rate = self.tax_withholding_rates.get(jurisdiction, 0.25)
        
        # Apply rate
        withholding = amount * Decimal(str(rate))
        
        # Check for tax treaty benefits
        if member.get("tax_treaty_benefits", False):
            withholding *= Decimal("0.5")  # 50% reduction for treaty benefits
        
        return withholding.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    
    def _get_payment_details(self, member: Dict[str, Any]) -> Dict[str, str]:
        """Get payment details for member"""
        payment_method = member.get("payment_method", "bank_transfer")
        
        if payment_method == "bank_transfer":
            return {
                "method": "bank_transfer",
                "account": member.get("bank_account", ""),
                "routing": member.get("routing_number", ""),
                "swift": member.get("swift_code", "")
            }
        elif payment_method == "crypto":
            return {
                "method": "crypto",
                "wallet_address": member.get("crypto_wallet", ""),
                "currency": member.get("crypto_currency", "USDC")
            }
        else:
            return {"method": payment_method}
    
    def _log_distribution(self, total_profit: Decimal, operational_reserve: Decimal,
                         distributions: Dict[str, Dict[str, Any]], model: DistributionModel):
        """Log distribution event"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "syndicate_id": self.syndicate_id,
            "total_profit": str(total_profit),
            "operational_reserve": str(operational_reserve),
            "distribution_model": model.value,
            "distributions": distributions,
            "total_distributed": str(sum(Decimal(d["gross_amount"]) for d in distributions.values()))
        }
        
        self.distribution_history.append(log_entry)
        
        # In production, this would persist to database
        print(f"Distribution logged: {json.dumps(log_entry, indent=2)}")
    
    def generate_distribution_report(self, distribution_id: str) -> Dict[str, Any]:
        """Generate detailed distribution report"""
        # Find distribution in history
        distribution = next((d for d in self.distribution_history 
                           if d.get("id") == distribution_id), None)
        
        if not distribution:
            raise ValueError(f"Distribution {distribution_id} not found")
        
        return {
            "summary": {
                "distribution_id": distribution_id,
                "date": distribution["timestamp"],
                "total_profit": distribution["total_profit"],
                "operational_reserve": distribution["operational_reserve"],
                "total_distributed": distribution["total_distributed"],
                "model_used": distribution["distribution_model"]
            },
            "member_details": distribution["distributions"],
            "tax_summary": self._generate_tax_summary(distribution["distributions"]),
            "payment_status": self._get_payment_status(distribution_id)
        }
    
    def _generate_tax_summary(self, distributions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate tax summary for distribution"""
        total_withheld = sum(Decimal(d["tax_withheld"]) for d in distributions.values())
        
        by_jurisdiction = {}
        for member_id, dist in distributions.items():
            # Would look up member jurisdiction
            jurisdiction = "US"  # Placeholder
            if jurisdiction not in by_jurisdiction:
                by_jurisdiction[jurisdiction] = Decimal("0")
            by_jurisdiction[jurisdiction] += Decimal(dist["tax_withheld"])
        
        return {
            "total_tax_withheld": str(total_withheld),
            "by_jurisdiction": {k: str(v) for k, v in by_jurisdiction.items()}
        }
    
    def _get_payment_status(self, distribution_id: str) -> Dict[str, str]:
        """Get payment status for distribution"""
        # Placeholder - would check actual payment processor
        return {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "method": "automated_transfer"
        }


class WithdrawalManager:
    """Manage member withdrawal requests"""
    
    def __init__(self, syndicate_id: str):
        self.syndicate_id = syndicate_id
        self.withdrawal_requests = []
        self.rules = {
            "minimum_notice_days": 7,
            "maximum_withdrawal_percentage": 0.50,
            "lockup_period_days": 90,
            "withdrawal_frequency": "monthly",
            "emergency_withdrawal_penalty": 0.10
        }
    
    def request_withdrawal(self, member_id: str, member_balance: Decimal, 
                         amount: Decimal, is_emergency: bool = False) -> Dict[str, Any]:
        """Process withdrawal request"""
        
        # Validate request
        validation = self._validate_withdrawal(member_id, member_balance, amount, is_emergency)
        
        if validation["status"] == "denied":
            return validation
        
        # Calculate actual withdrawal amount and fees
        if is_emergency:
            penalty = amount * Decimal(str(self.rules["emergency_withdrawal_penalty"]))
            net_amount = amount - penalty
        else:
            net_amount = amount
            penalty = Decimal("0")
        
        # Schedule withdrawal
        if is_emergency:
            scheduled_date = datetime.now() + timedelta(days=1)
        else:
            scheduled_date = datetime.now() + timedelta(days=self.rules["minimum_notice_days"])
        
        request = {
            "id": str(uuid.uuid4()),
            "member_id": member_id,
            "requested_amount": str(amount),
            "approved_amount": str(validation.get("approved_amount", amount)),
            "penalty": str(penalty),
            "net_amount": str(net_amount),
            "is_emergency": is_emergency,
            "status": "scheduled",
            "requested_at": datetime.now().isoformat(),
            "scheduled_for": scheduled_date.isoformat(),
            "voting_power_impact": self._calculate_voting_impact(member_balance, amount)
        }
        
        self.withdrawal_requests.append(request)
        
        return request
    
    def _validate_withdrawal(self, member_id: str, member_balance: Decimal, 
                           amount: Decimal, is_emergency: bool) -> Dict[str, Any]:
        """Validate withdrawal request"""
        
        # Check if member exists and is active
        # Placeholder - would check actual member status
        
        # Check lockup period
        # Placeholder - would check member join date
        days_since_joining = 100  # Placeholder
        
        if days_since_joining < self.rules["lockup_period_days"] and not is_emergency:
            return {
                "status": "denied",
                "reason": f"Lockup period active ({self.rules['lockup_period_days']} days)",
                "days_remaining": self.rules["lockup_period_days"] - days_since_joining
            }
        
        # Check withdrawal limits
        max_allowed = member_balance * Decimal(str(self.rules["maximum_withdrawal_percentage"]))
        
        if amount > max_allowed:
            return {
                "status": "partial_approval",
                "approved_amount": max_allowed,
                "reason": f"Exceeds maximum withdrawal percentage ({self.rules['maximum_withdrawal_percentage']*100}%)"
            }
        
        # Check minimum balance requirements
        remaining_balance = member_balance - amount
        min_balance = Decimal("100")  # Minimum $100 balance
        
        if remaining_balance < min_balance:
            return {
                "status": "partial_approval",
                "approved_amount": member_balance - min_balance,
                "reason": f"Must maintain minimum balance of ${min_balance}"
            }
        
        return {"status": "approved", "approved_amount": amount}
    
    def _calculate_voting_impact(self, current_balance: Decimal, withdrawal_amount: Decimal) -> Dict[str, float]:
        """Calculate impact on member's voting power"""
        new_balance = current_balance - withdrawal_amount
        
        # Simplified calculation
        current_power = float(current_balance / Decimal("100000"))  # Normalized
        new_power = float(new_balance / Decimal("100000"))
        
        return {
            "current_voting_power": current_power,
            "new_voting_power": new_power,
            "power_reduction_percentage": (current_power - new_power) / current_power * 100 if current_power > 0 else 0
        }
    
    def process_scheduled_withdrawals(self) -> List[Dict[str, Any]]:
        """Process withdrawals scheduled for today"""
        processed = []
        today = datetime.now().date()
        
        for request in self.withdrawal_requests:
            if request["status"] == "scheduled":
                scheduled_date = datetime.fromisoformat(request["scheduled_for"]).date()
                
                if scheduled_date <= today:
                    # Process withdrawal
                    request["status"] = "processing"
                    request["processed_at"] = datetime.now().isoformat()
                    
                    # Initiate actual transfer (placeholder)
                    transfer_result = self._initiate_transfer(request)
                    
                    if transfer_result["success"]:
                        request["status"] = "completed"
                        request["transaction_id"] = transfer_result["transaction_id"]
                    else:
                        request["status"] = "failed"
                        request["failure_reason"] = transfer_result["reason"]
                    
                    processed.append(request)
        
        return processed
    
    def _initiate_transfer(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Initiate actual fund transfer"""
        # Placeholder for actual payment processing
        return {
            "success": True,
            "transaction_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat()
        }