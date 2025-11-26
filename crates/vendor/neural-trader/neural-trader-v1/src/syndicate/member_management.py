"""
Syndicate Member Management System
Handles member roles, permissions, and performance tracking
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import uuid
import json
from decimal import Decimal


class MemberRole(Enum):
    """Member role definitions"""
    LEAD_INVESTOR = "lead_investor"
    SENIOR_ANALYST = "senior_analyst"
    JUNIOR_ANALYST = "junior_analyst"
    CONTRIBUTING_MEMBER = "contributing_member"
    OBSERVER = "observer"


class MemberTier(Enum):
    """Investment tier definitions"""
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"


@dataclass
class MemberPermissions:
    """Member permission configuration"""
    create_syndicate: bool = False
    modify_strategy: bool = False
    approve_large_bets: bool = False
    manage_members: bool = False
    distribute_profits: bool = False
    access_all_analytics: bool = False
    veto_power: bool = False
    propose_bets: bool = False
    access_advanced_analytics: bool = False
    create_models: bool = False
    vote_on_strategy: bool = False
    manage_junior_analysts: bool = False
    view_bets: bool = True
    vote_on_major_decisions: bool = False
    access_basic_analytics: bool = True
    propose_ideas: bool = True
    withdraw_own_funds: bool = True


@dataclass
class InvestmentTierConfig:
    """Investment tier configuration"""
    min_investment: Decimal
    max_investment: Optional[Decimal]
    profit_share: float
    voting_weight_multiplier: float
    features: List[str]


class Member:
    """Syndicate member representation"""
    
    def __init__(self, member_id: str, name: str, email: str, role: MemberRole):
        self.id = member_id
        self.name = name
        self.email = email
        self.role = role
        self.joined_date = datetime.now()
        self.capital_contribution = Decimal("0")
        self.performance_score = 0.0
        self.roi_score = 0.0
        self.accuracy_score = 0.0
        self.is_active = True
        self.tier = MemberTier.BRONZE
        self.permissions = self._get_role_permissions(role)
        self.stats = MemberStatistics()
        
    def _get_role_permissions(self, role: MemberRole) -> MemberPermissions:
        """Get permissions based on role"""
        permissions_map = {
            MemberRole.LEAD_INVESTOR: MemberPermissions(
                create_syndicate=True,
                modify_strategy=True,
                approve_large_bets=True,
                manage_members=True,
                distribute_profits=True,
                access_all_analytics=True,
                veto_power=True,
                propose_bets=True,
                vote_on_strategy=True,
                vote_on_major_decisions=True
            ),
            MemberRole.SENIOR_ANALYST: MemberPermissions(
                propose_bets=True,
                access_advanced_analytics=True,
                create_models=True,
                vote_on_strategy=True,
                manage_junior_analysts=True,
                vote_on_major_decisions=True
            ),
            MemberRole.JUNIOR_ANALYST: MemberPermissions(
                propose_bets=True,
                access_advanced_analytics=True,
                vote_on_strategy=False,
                vote_on_major_decisions=False
            ),
            MemberRole.CONTRIBUTING_MEMBER: MemberPermissions(
                vote_on_major_decisions=True,
                access_basic_analytics=True,
                propose_ideas=True,
                withdraw_own_funds=True
            ),
            MemberRole.OBSERVER: MemberPermissions(
                view_bets=True,
                access_basic_analytics=False,
                propose_ideas=False,
                withdraw_own_funds=False
            )
        }
        return permissions_map.get(role, MemberPermissions())
    
    def update_tier(self, new_contribution: Decimal):
        """Update member tier based on contribution"""
        self.capital_contribution = new_contribution
        
        tier_thresholds = {
            MemberTier.BRONZE: (Decimal("1000"), Decimal("5000")),
            MemberTier.SILVER: (Decimal("5000"), Decimal("25000")),
            MemberTier.GOLD: (Decimal("25000"), Decimal("100000")),
            MemberTier.PLATINUM: (Decimal("100000"), None)
        }
        
        for tier, (min_val, max_val) in tier_thresholds.items():
            if max_val is None and new_contribution >= min_val:
                self.tier = tier
                break
            elif min_val <= new_contribution < max_val:
                self.tier = tier
                break
    
    def calculate_voting_weight(self, syndicate_total_capital: Decimal) -> float:
        """Calculate member's voting weight"""
        if not self.is_active:
            return 0.0
        
        # Base weight on capital contribution (50%)
        capital_weight = float(self.capital_contribution / syndicate_total_capital) * 0.5
        
        # Performance weight (30%)
        performance_weight = self.performance_score * 0.3
        
        # Tenure weight (20%)
        months_active = (datetime.now() - self.joined_date).days / 30
        tenure_weight = min(months_active / 12, 1.0) * 0.2
        
        # Role multiplier
        role_multipliers = {
            MemberRole.LEAD_INVESTOR: 1.5,
            MemberRole.SENIOR_ANALYST: 1.3,
            MemberRole.JUNIOR_ANALYST: 1.1,
            MemberRole.CONTRIBUTING_MEMBER: 1.0,
            MemberRole.OBSERVER: 0.0
        }
        
        base_weight = capital_weight + performance_weight + tenure_weight
        return base_weight * role_multipliers.get(self.role, 1.0)


@dataclass
class MemberStatistics:
    """Track member performance statistics"""
    bets_proposed: int = 0
    bets_approved: int = 0
    bets_won: int = 0
    bets_lost: int = 0
    total_profit: Decimal = Decimal("0")
    total_staked: Decimal = Decimal("0")
    best_bet_profit: Decimal = Decimal("0")
    worst_bet_loss: Decimal = Decimal("0")
    specialties: List[str] = None
    win_rate_by_sport: Dict[str, float] = None
    
    def __post_init__(self):
        if self.specialties is None:
            self.specialties = []
        if self.win_rate_by_sport is None:
            self.win_rate_by_sport = {}
    
    @property
    def win_rate(self) -> float:
        """Calculate overall win rate"""
        total_bets = self.bets_won + self.bets_lost
        return self.bets_won / total_bets if total_bets > 0 else 0.0
    
    @property
    def roi(self) -> float:
        """Calculate return on investment"""
        if self.total_staked == 0:
            return 0.0
        return float((self.total_profit / self.total_staked) * 100)


class MemberPerformanceTracker:
    """Track and analyze member performance"""
    
    def __init__(self):
        self.performance_history = {}
        self.skill_assessments = {}
        
    def track_bet_outcome(self, member_id: str, bet_details: Dict[str, Any]):
        """Track the outcome of a member's bet"""
        if member_id not in self.performance_history:
            self.performance_history[member_id] = []
        
        performance_record = {
            "timestamp": datetime.now(),
            "bet_id": bet_details["bet_id"],
            "sport": bet_details["sport"],
            "bet_type": bet_details["bet_type"],
            "odds": bet_details["odds"],
            "stake": bet_details["stake"],
            "outcome": bet_details["outcome"],
            "profit": bet_details["profit"],
            "confidence": bet_details.get("confidence", 0.5),
            "edge": bet_details.get("edge", 0.0),
            "reasoning_quality": self._assess_reasoning(bet_details.get("reasoning", ""))
        }
        
        self.performance_history[member_id].append(performance_record)
        self._update_skill_assessment(member_id, performance_record)
    
    def _assess_reasoning(self, reasoning: str) -> float:
        """Assess quality of betting reasoning"""
        # Simplified assessment based on key factors
        quality_indicators = [
            "statistical", "trend", "value", "edge", "probability",
            "analysis", "model", "data", "historical", "pattern"
        ]
        
        reasoning_lower = reasoning.lower()
        indicator_count = sum(1 for indicator in quality_indicators if indicator in reasoning_lower)
        
        # Score based on presence of analytical terms and length
        base_score = min(indicator_count / len(quality_indicators), 1.0)
        length_factor = min(len(reasoning) / 500, 1.0)  # Expect ~500 chars for good reasoning
        
        return (base_score * 0.7 + length_factor * 0.3)
    
    def _update_skill_assessment(self, member_id: str, performance: Dict[str, Any]):
        """Update member's skill assessment"""
        if member_id not in self.skill_assessments:
            self.skill_assessments[member_id] = {
                "sports": {},
                "bet_types": {},
                "odds_ranges": {},
                "overall_skill": 0.5
            }
        
        assessment = self.skill_assessments[member_id]
        
        # Update sport-specific skill
        sport = performance["sport"]
        if sport not in assessment["sports"]:
            assessment["sports"][sport] = {"attempts": 0, "success_rate": 0.0, "roi": 0.0}
        
        sport_stats = assessment["sports"][sport]
        sport_stats["attempts"] += 1
        
        # Update success rate with exponential moving average
        alpha = 0.1  # Learning rate
        success = 1.0 if performance["profit"] > 0 else 0.0
        sport_stats["success_rate"] = (1 - alpha) * sport_stats["success_rate"] + alpha * success
        
        # Update ROI
        roi = float(performance["profit"] / performance["stake"]) if performance["stake"] > 0 else 0.0
        sport_stats["roi"] = (1 - alpha) * sport_stats["roi"] + alpha * roi
    
    def calculate_member_alpha(self, member_id: str) -> Dict[str, float]:
        """Calculate member's alpha (skill-based returns)"""
        if member_id not in self.performance_history:
            return {"alpha": 0.0, "confidence": 0.0}
        
        member_bets = self.performance_history[member_id]
        
        if len(member_bets) < 10:  # Need minimum sample size
            return {"alpha": 0.0, "confidence": 0.1}
        
        # Calculate actual returns
        total_staked = sum(bet["stake"] for bet in member_bets)
        total_profit = sum(bet["profit"] for bet in member_bets)
        actual_roi = float(total_profit / total_staked) if total_staked > 0 else 0.0
        
        # Calculate expected returns based on odds (assuming efficient market)
        expected_returns = []
        for bet in member_bets:
            fair_probability = 1.0 / bet["odds"]
            expected_return = -bet["stake"]  # Assume negative EV at market odds
            expected_returns.append(expected_return)
        
        expected_total = sum(expected_returns)
        expected_roi = float(expected_total / total_staked) if total_staked > 0 else 0.0
        
        # Alpha is the excess return
        alpha = actual_roi - expected_roi
        
        # Confidence based on sample size and consistency
        sample_confidence = min(len(member_bets) / 100, 1.0)  # Max confidence at 100 bets
        
        # Calculate consistency (lower variance = higher confidence)
        if len(member_bets) > 1:
            returns = [float(bet["profit"] / bet["stake"]) for bet in member_bets if bet["stake"] > 0]
            if returns:
                import statistics
                try:
                    variance = statistics.variance(returns)
                    consistency_score = 1.0 / (1.0 + variance)  # Higher consistency for lower variance
                except:
                    consistency_score = 0.5
            else:
                consistency_score = 0.5
        else:
            consistency_score = 0.5
        
        confidence = sample_confidence * consistency_score
        
        return {
            "alpha": alpha,
            "confidence": confidence,
            "sample_size": len(member_bets),
            "consistency_score": consistency_score
        }
    
    def identify_member_strengths(self, member_id: str) -> Dict[str, Any]:
        """Identify member's betting strengths"""
        if member_id not in self.skill_assessments:
            return {"strengths": [], "weaknesses": [], "recommendations": []}
        
        assessment = self.skill_assessments[member_id]
        strengths = []
        weaknesses = []
        recommendations = []
        
        # Analyze sport performance
        for sport, stats in assessment["sports"].items():
            if stats["attempts"] >= 5:  # Minimum sample
                if stats["success_rate"] > 0.55 and stats["roi"] > 0.05:
                    strengths.append(f"Strong performance in {sport} betting")
                elif stats["success_rate"] < 0.45 or stats["roi"] < -0.10:
                    weaknesses.append(f"Underperforming in {sport} betting")
        
        # Analyze bet types
        for bet_type, stats in assessment.get("bet_types", {}).items():
            if stats["attempts"] >= 5:
                if stats["success_rate"] > 0.55:
                    strengths.append(f"Skilled at {bet_type} bets")
                elif stats["success_rate"] < 0.45:
                    weaknesses.append(f"Struggling with {bet_type} bets")
        
        # Generate recommendations
        if strengths:
            recommendations.append(f"Focus on your strengths: {', '.join(strengths[:2])}")
        
        if weaknesses:
            recommendations.append(f"Consider reducing exposure to: {', '.join(weaknesses[:2])}")
        
        # Check for needed diversification
        if len(assessment["sports"]) < 3:
            recommendations.append("Consider diversifying into more sports")
        
        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations,
            "overall_assessment": self._generate_overall_assessment(assessment)
        }
    
    def _generate_overall_assessment(self, assessment: Dict[str, Any]) -> str:
        """Generate overall assessment narrative"""
        total_sports = len(assessment["sports"])
        avg_success_rate = sum(s["success_rate"] for s in assessment["sports"].values()) / total_sports if total_sports > 0 else 0.0
        
        if avg_success_rate > 0.55:
            return "Consistently profitable bettor with strong analytical skills"
        elif avg_success_rate > 0.50:
            return "Solid performer with room for improvement"
        elif avg_success_rate > 0.45:
            return "Average performance - focus on identified strengths"
        else:
            return "Needs significant improvement - consider education and mentoring"


class SyndicateMemberManager:
    """Manage syndicate members"""
    
    def __init__(self, syndicate_id: str):
        self.syndicate_id = syndicate_id
        self.members: Dict[str, Member] = {}
        self.performance_tracker = MemberPerformanceTracker()
        self.voting_system = VotingSystem(self)
        
    def add_member(self, name: str, email: str, role: MemberRole, initial_contribution: Decimal) -> Member:
        """Add new member to syndicate"""
        member_id = str(uuid.uuid4())
        member = Member(member_id, name, email, role)
        member.update_tier(initial_contribution)
        
        self.members[member_id] = member
        
        # Log the addition
        self._log_member_action("add_member", member_id, {
            "name": name,
            "role": role.value,
            "initial_contribution": str(initial_contribution)
        })
        
        return member
    
    def update_member_role(self, member_id: str, new_role: MemberRole, authorized_by: str):
        """Update member's role"""
        if member_id not in self.members:
            raise ValueError(f"Member {member_id} not found")
        
        # Check authorization
        authorizer = self.members.get(authorized_by)
        if not authorizer or not authorizer.permissions.manage_members:
            raise PermissionError(f"Member {authorized_by} not authorized to manage members")
        
        old_role = self.members[member_id].role
        self.members[member_id].role = new_role
        self.members[member_id].permissions = self.members[member_id]._get_role_permissions(new_role)
        
        self._log_member_action("update_role", member_id, {
            "old_role": old_role.value,
            "new_role": new_role.value,
            "authorized_by": authorized_by
        })
    
    def suspend_member(self, member_id: str, reason: str, authorized_by: str):
        """Suspend a member"""
        if member_id not in self.members:
            raise ValueError(f"Member {member_id} not found")
        
        # Check authorization
        authorizer = self.members.get(authorized_by)
        if not authorizer or not authorizer.permissions.manage_members:
            raise PermissionError(f"Member {authorized_by} not authorized to manage members")
        
        self.members[member_id].is_active = False
        
        self._log_member_action("suspend_member", member_id, {
            "reason": reason,
            "authorized_by": authorized_by
        })
    
    def get_member_performance_report(self, member_id: str) -> Dict[str, Any]:
        """Generate comprehensive performance report for member"""
        if member_id not in self.members:
            raise ValueError(f"Member {member_id} not found")
        
        member = self.members[member_id]
        
        return {
            "member_info": {
                "id": member.id,
                "name": member.name,
                "role": member.role.value,
                "tier": member.tier.value,
                "joined_date": member.joined_date.isoformat(),
                "is_active": member.is_active
            },
            "financial_summary": {
                "capital_contribution": str(member.capital_contribution),
                "total_profit": str(member.stats.total_profit),
                "roi": member.stats.roi,
                "profit_share_percentage": self._get_profit_share_percentage(member)
            },
            "betting_performance": {
                "total_bets": member.stats.bets_proposed,
                "approved_bets": member.stats.bets_approved,
                "win_rate": member.stats.win_rate,
                "best_bet_profit": str(member.stats.best_bet_profit),
                "worst_bet_loss": str(member.stats.worst_bet_loss),
                "win_rate_by_sport": member.stats.win_rate_by_sport
            },
            "skill_assessment": self.performance_tracker.identify_member_strengths(member_id),
            "alpha_analysis": self.performance_tracker.calculate_member_alpha(member_id),
            "voting_weight": member.calculate_voting_weight(self.get_total_capital()),
            "recent_activity": self._get_recent_activity(member_id)
        }
    
    def _get_profit_share_percentage(self, member: Member) -> float:
        """Calculate member's profit share percentage"""
        tier_shares = {
            MemberTier.BRONZE: 0.80,
            MemberTier.SILVER: 0.85,
            MemberTier.GOLD: 0.90,
            MemberTier.PLATINUM: 0.95
        }
        return tier_shares.get(member.tier, 0.80)
    
    def get_total_capital(self) -> Decimal:
        """Get total syndicate capital"""
        return sum(member.capital_contribution for member in self.members.values() if member.is_active)
    
    def _get_recent_activity(self, member_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get member's recent activity"""
        # This would query from activity log
        # Placeholder implementation
        return []
    
    def _log_member_action(self, action: str, member_id: str, details: Dict[str, Any]):
        """Log member-related actions"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "syndicate_id": self.syndicate_id,
            "action": action,
            "member_id": member_id,
            "details": details
        }
        # In production, this would write to a persistent log
        print(f"Member action logged: {json.dumps(log_entry, indent=2)}")


class VotingSystem:
    """Handle syndicate voting mechanisms"""
    
    def __init__(self, member_manager: SyndicateMemberManager):
        self.member_manager = member_manager
        self.active_votes = {}
        
    def create_vote(self, proposal_type: str, proposal_details: Dict[str, Any], 
                    proposed_by: str, voting_period_hours: int = 24) -> str:
        """Create a new vote"""
        vote_id = str(uuid.uuid4())
        
        self.active_votes[vote_id] = {
            "id": vote_id,
            "type": proposal_type,
            "details": proposal_details,
            "proposed_by": proposed_by,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(hours=voting_period_hours),
            "votes": {},
            "status": "active"
        }
        
        return vote_id
    
    def cast_vote(self, vote_id: str, member_id: str, decision: str) -> bool:
        """Cast a vote"""
        if vote_id not in self.active_votes:
            raise ValueError(f"Vote {vote_id} not found")
        
        vote = self.active_votes[vote_id]
        
        if vote["status"] != "active":
            raise ValueError(f"Vote {vote_id} is not active")
        
        if datetime.now() > vote["expires_at"]:
            vote["status"] = "expired"
            raise ValueError(f"Vote {vote_id} has expired")
        
        member = self.member_manager.members.get(member_id)
        if not member or not member.is_active:
            raise ValueError(f"Member {member_id} not eligible to vote")
        
        # Calculate voting weight
        weight = member.calculate_voting_weight(self.member_manager.get_total_capital())
        
        vote["votes"][member_id] = {
            "decision": decision,
            "weight": weight,
            "timestamp": datetime.now()
        }
        
        return True
    
    def get_vote_results(self, vote_id: str) -> Dict[str, Any]:
        """Get current vote results"""
        if vote_id not in self.active_votes:
            raise ValueError(f"Vote {vote_id} not found")
        
        vote = self.active_votes[vote_id]
        
        # Tally votes
        results = {"approve": 0.0, "reject": 0.0, "abstain": 0.0}
        
        for member_vote in vote["votes"].values():
            results[member_vote["decision"]] += member_vote["weight"]
        
        total_weight = sum(results.values())
        
        return {
            "vote_id": vote_id,
            "status": vote["status"],
            "results": results,
            "total_votes": len(vote["votes"]),
            "total_weight": total_weight,
            "approval_percentage": (results["approve"] / total_weight * 100) if total_weight > 0 else 0,
            "participation_rate": len(vote["votes"]) / len(self.member_manager.members) * 100
        }