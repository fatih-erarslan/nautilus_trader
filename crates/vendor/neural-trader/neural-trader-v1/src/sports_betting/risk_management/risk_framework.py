"""
Integrated Risk Management Framework for Sports Betting

Provides a unified interface for all risk management components
in a sports betting syndicate operation.
"""

import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import json

from .portfolio_risk import (
    PortfolioRiskManager, BetOpportunity, PortfolioAllocation,
    BettingStrategy
)
from .betting_limits import (
    BettingLimitsController, BetRecord, CircuitBreakerStatus
)
from .market_risk import (
    MarketRiskAnalyzer, MarketRiskAssessment, BookmakerProfile,
    RegulatoryAlert, RiskLevel
)
from .syndicate_risk import (
    SyndicateRiskController, SyndicateMember, BettingProposal,
    MemberRole, ExpertiseLevel, EmergencyStatus
)
from .performance_monitor import (
    PerformanceMonitor, BettingTransaction, TimeFrame,
    PerformanceMetrics
)


logger = logging.getLogger(__name__)


@dataclass
class RiskDecision:
    """Risk management decision for a betting opportunity"""
    bet_id: str
    approved: bool
    allocated_amount: float
    risk_score: float
    violations: List[str]
    warnings: List[str]
    recommendations: List[str]
    member_allocations: Dict[str, float]
    required_consensus: bool
    proposal_id: Optional[str] = None


@dataclass
class SystemHealthCheck:
    """System health check results"""
    timestamp: datetime.datetime
    overall_status: str  # 'healthy', 'warning', 'critical'
    components: Dict[str, str]
    metrics: Dict[str, Any]
    issues: List[str]
    recommendations: List[str]


class RiskFramework:
    """
    Integrated risk management framework for sports betting syndicate operations.
    Coordinates all risk management components and provides unified decision-making.
    """
    
    def __init__(self,
                 syndicate_name: str,
                 initial_bankroll: float,
                 config: Optional[Dict] = None):
        """
        Initialize Risk Management Framework
        
        Args:
            syndicate_name: Name of the betting syndicate
            initial_bankroll: Total initial bankroll
            config: Optional configuration dictionary
        """
        self.syndicate_name = syndicate_name
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        
        # Default configuration
        default_config = {
            'max_kelly_fraction': 0.25,
            'max_portfolio_risk': 0.10,
            'max_bet_percentage': 0.05,
            'max_daily_loss_percentage': 0.10,
            'max_drawdown_percentage': 0.20,
            'max_odds_volatility': 0.10,
            'min_liquidity_score': 0.7,
            'max_bookmaker_exposure': 50000,
            'max_member_allocation': 0.20,
            'risk_free_rate': 0.02,
            'target_return': 0.20
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Initialize components
        self.portfolio_manager = PortfolioRiskManager(
            bankroll=initial_bankroll,
            max_kelly_fraction=self.config['max_kelly_fraction'],
            max_portfolio_risk=self.config['max_portfolio_risk']
        )
        
        self.limits_controller = BettingLimitsController(
            bankroll=initial_bankroll,
            max_bet_percentage=self.config['max_bet_percentage'],
            max_daily_loss_percentage=self.config['max_daily_loss_percentage'],
            max_drawdown_percentage=self.config['max_drawdown_percentage']
        )
        
        self.market_analyzer = MarketRiskAnalyzer(
            max_odds_volatility=self.config['max_odds_volatility'],
            min_liquidity_score=self.config['min_liquidity_score'],
            max_bookmaker_exposure=self.config['max_bookmaker_exposure']
        )
        
        self.syndicate_controller = SyndicateRiskController(
            syndicate_name=syndicate_name,
            total_bankroll=initial_bankroll,
            max_member_allocation=self.config['max_member_allocation']
        )
        
        self.performance_monitor = PerformanceMonitor(
            initial_bankroll=initial_bankroll,
            risk_free_rate=self.config['risk_free_rate'],
            target_return=self.config['target_return'],
            max_drawdown_threshold=self.config['max_drawdown_percentage']
        )
        
        # Decision tracking
        self.decision_history: List[RiskDecision] = []
        self.active_bets: Dict[str, Dict] = {}
        
    def evaluate_betting_opportunity(self,
                                     bet_opportunity: BetOpportunity,
                                     bookmaker: str,
                                     jurisdiction: str,
                                     proposer_id: Optional[str] = None,
                                     participating_members: Optional[List[str]] = None
                                     ) -> RiskDecision:
        """
        Comprehensive evaluation of a betting opportunity
        
        Args:
            bet_opportunity: The betting opportunity to evaluate
            bookmaker: Bookmaker name
            jurisdiction: Betting jurisdiction
            proposer_id: ID of member proposing the bet
            participating_members: List of participating member IDs
            
        Returns:
            RiskDecision with approval status and details
        """
        logger.info(f"Evaluating bet opportunity: {bet_opportunity.bet_id}")
        
        violations = []
        warnings = []
        recommendations = []
        
        # 1. Portfolio Risk Analysis
        portfolio_allocations = self.portfolio_manager.optimize_multi_sport_portfolio(
            [bet_opportunity]
        )
        
        if not portfolio_allocations:
            violations.append("Portfolio optimization failed - insufficient edge or high correlation")
            
        recommended_allocation = (
            portfolio_allocations[0].allocation_percentage if portfolio_allocations else 0
        )
        
        # 2. Calculate stake amount
        proposed_stake = self.current_bankroll * recommended_allocation
        
        # 3. Check Betting Limits
        limits_ok, limit_violations = self.limits_controller.check_bet_limits(
            proposed_stake,
            bet_opportunity.sport,
            sum(self.active_bets.values()) if self.active_bets else 0
        )
        
        if not limits_ok:
            violations.extend(limit_violations)
            
        # 4. Check Circuit Breakers
        breakers_ok, triggered_breakers = self.limits_controller.check_circuit_breakers()
        
        if not breakers_ok:
            violations.extend(triggered_breakers)
            
        # 5. Market Risk Assessment
        market_assessment = self.market_analyzer.perform_comprehensive_risk_assessment(
            market_id=bet_opportunity.bet_id,
            bookmaker=bookmaker,
            jurisdiction=jurisdiction,
            sport=bet_opportunity.sport,
            proposed_stake=proposed_stake
        )
        
        if market_assessment.overall_risk == RiskLevel.HIGH:
            warnings.append(f"High market risk: {', '.join(market_assessment.risk_factors)}")
        elif market_assessment.overall_risk == RiskLevel.CRITICAL:
            violations.append(f"Critical market risk: {', '.join(market_assessment.risk_factors)}")
            
        recommendations.extend(market_assessment.recommendations)
        
        # 6. Syndicate Controls
        requires_consensus = proposed_stake > 1000  # Threshold for consensus
        member_allocations = {}
        proposal_id = None
        
        if requires_consensus and proposer_id:
            # Create betting proposal
            proposal = self.syndicate_controller.create_betting_proposal(
                proposer_id=proposer_id,
                sport=bet_opportunity.sport,
                event=bet_opportunity.event,
                selection=bet_opportunity.selection,
                odds=bet_opportunity.odds,
                proposed_stake=proposed_stake,
                rationale=f"Edge: {bet_opportunity.edge:.2%}, Confidence: {bet_opportunity.confidence:.1f}"
            )
            
            if proposal:
                proposal_id = proposal.proposal_id
                warnings.append(f"Consensus required - Proposal {proposal_id} created")
            else:
                violations.append("Failed to create betting proposal")
                
        # 7. Member Allocation
        if participating_members:
            member_allocations = self.syndicate_controller.allocate_risk_by_expertise(
                total_stake=proposed_stake,
                sport=bet_opportunity.sport,
                participating_members=participating_members
            )
            
            # Check member limits
            for member_id, allocation in member_allocations.items():
                member_ok, member_violations = self.syndicate_controller.check_member_limit(
                    member_id, allocation, bet_opportunity.sport
                )
                if not member_ok:
                    warnings.extend([f"Member {member_id}: {v}" for v in member_violations])
                    
        # 8. Calculate risk score
        risk_components = {
            'portfolio_risk': 1 - recommended_allocation / self.config['max_portfolio_risk'],
            'limit_utilization': len(limit_violations) * 0.2,
            'market_risk': {
                RiskLevel.LOW: 0,
                RiskLevel.MEDIUM: 0.3,
                RiskLevel.HIGH: 0.6,
                RiskLevel.CRITICAL: 1.0
            }[market_assessment.overall_risk],
            'circuit_breaker': 0.5 if not breakers_ok else 0
        }
        
        risk_score = sum(risk_components.values()) / len(risk_components)
        
        # 9. Make decision
        approved = (
            len(violations) == 0 and
            risk_score < 0.7 and
            (not requires_consensus or proposal_id is not None)
        )
        
        # Create decision record
        decision = RiskDecision(
            bet_id=bet_opportunity.bet_id,
            approved=approved,
            allocated_amount=proposed_stake if approved else 0,
            risk_score=risk_score,
            violations=violations,
            warnings=warnings,
            recommendations=recommendations,
            member_allocations=member_allocations,
            required_consensus=requires_consensus,
            proposal_id=proposal_id
        )
        
        # Track decision
        self.decision_history.append(decision)
        
        logger.info(
            f"Risk decision for {bet_opportunity.bet_id}: "
            f"{'APPROVED' if approved else 'REJECTED'} "
            f"(risk score: {risk_score:.2f})"
        )
        
        return decision
        
    def place_bet(self,
                  bet_opportunity: BetOpportunity,
                  decision: RiskDecision,
                  bookmaker: str) -> bool:
        """
        Place a bet after risk approval
        
        Args:
            bet_opportunity: The betting opportunity
            decision: Risk decision
            bookmaker: Bookmaker to place with
            
        Returns:
            True if bet placed successfully
        """
        if not decision.approved:
            logger.warning(f"Attempting to place unapproved bet {bet_opportunity.bet_id}")
            return False
            
        # Record bet with limits controller
        bet_record = BetRecord(
            bet_id=bet_opportunity.bet_id,
            sport=bet_opportunity.sport,
            amount=decision.allocated_amount,
            timestamp=datetime.datetime.now(),
            odds=bet_opportunity.odds
        )
        
        self.limits_controller.record_bet(bet_record)
        
        # Update bookmaker exposure
        self.market_analyzer.update_bookmaker_exposure(
            bookmaker, decision.allocated_amount
        )
        
        # Create transaction for performance monitoring
        transaction = BettingTransaction(
            transaction_id=bet_opportunity.bet_id,
            timestamp=datetime.datetime.now(),
            sport=bet_opportunity.sport,
            event=bet_opportunity.event,
            selection=bet_opportunity.selection,
            bet_type='single',  # Could be expanded
            stake=decision.allocated_amount,
            odds=bet_opportunity.odds
        )
        
        self.performance_monitor.record_transaction(transaction)
        
        # Track active bet
        self.active_bets[bet_opportunity.bet_id] = {
            'amount': decision.allocated_amount,
            'odds': bet_opportunity.odds,
            'timestamp': datetime.datetime.now()
        }
        
        logger.info(
            f"Bet placed: {bet_opportunity.bet_id} - "
            f"${decision.allocated_amount:.2f} at {bet_opportunity.odds}"
        )
        
        return True
        
    def update_bet_result(self,
                          bet_id: str,
                          result: str,
                          settlement_amount: float) -> bool:
        """
        Update the result of a bet
        
        Args:
            bet_id: Bet identifier
            result: Result ('win', 'loss', 'push', 'void')
            settlement_amount: Amount won/lost
            
        Returns:
            True if updated successfully
        """
        if bet_id not in self.active_bets:
            logger.warning(f"Bet {bet_id} not found in active bets")
            return False
            
        # Calculate P&L
        stake = self.active_bets[bet_id]['amount']
        if result == 'win':
            pnl = settlement_amount - stake
        elif result == 'loss':
            pnl = -stake
        else:  # push or void
            pnl = 0
            
        # Update limits controller
        self.limits_controller.update_bet_result(bet_id, result, pnl)
        
        # Update performance monitor
        self.performance_monitor.update_transaction_result(bet_id, result, pnl)
        
        # Update bankroll
        self.current_bankroll += pnl
        self.portfolio_manager.update_bankroll(self.current_bankroll)
        
        # Remove from active bets
        del self.active_bets[bet_id]
        
        logger.info(f"Bet {bet_id} settled: {result} - P&L: ${pnl:.2f}")
        
        return True
        
    def perform_health_check(self) -> SystemHealthCheck:
        """
        Perform comprehensive system health check
        
        Returns:
            SystemHealthCheck with status and recommendations
        """
        health_check = SystemHealthCheck(
            timestamp=datetime.datetime.now(),
            overall_status='healthy',
            components={},
            metrics={},
            issues=[],
            recommendations=[]
        )
        
        # Check circuit breakers
        breakers_ok, triggered = self.limits_controller.check_circuit_breakers()
        health_check.components['circuit_breakers'] = 'healthy' if breakers_ok else 'critical'
        if not breakers_ok:
            health_check.issues.extend(triggered)
            
        # Check drawdown
        action, factor = self.limits_controller.check_drawdown_limits()
        health_check.components['drawdown_control'] = {
            'continue': 'healthy',
            'alert': 'warning',
            'reduce': 'warning',
            'stop': 'critical'
        }.get(action, 'healthy')
        
        if action != 'continue':
            health_check.issues.append(f"Drawdown action triggered: {action}")
            
        # Check emergency status
        emergency_status, emergency_issues = self.syndicate_controller.check_emergency_conditions()
        health_check.components['emergency_status'] = {
            EmergencyStatus.NORMAL: 'healthy',
            EmergencyStatus.ALERT: 'warning',
            EmergencyStatus.WARNING: 'warning',
            EmergencyStatus.EMERGENCY: 'critical',
            EmergencyStatus.SHUTDOWN: 'critical'
        }[emergency_status]
        
        if emergency_issues:
            health_check.issues.extend(emergency_issues)
            
        # Get performance metrics
        metrics = self.performance_monitor.get_performance_metrics()
        health_check.metrics = {
            'total_pnl': metrics.total_pnl,
            'win_rate': metrics.win_rate,
            'sharpe_ratio': metrics.sharpe_ratio,
            'current_drawdown': metrics.current_drawdown,
            'consecutive_losses': metrics.consecutive_losses
        }
        
        # Check performance alerts
        active_alerts = [
            a for a in self.performance_monitor.active_alerts.values()
            if not a.resolved
        ]
        
        health_check.components['performance'] = (
            'healthy' if not active_alerts else
            'warning' if len(active_alerts) < 3 else
            'critical'
        )
        
        for alert in active_alerts:
            health_check.issues.append(f"{alert.alert_type.value}: {alert.message}")
            
        # Determine overall status
        component_statuses = list(health_check.components.values())
        if 'critical' in component_statuses:
            health_check.overall_status = 'critical'
        elif 'warning' in component_statuses:
            health_check.overall_status = 'warning'
            
        # Generate recommendations
        if metrics.sharpe_ratio < 0.5:
            health_check.recommendations.append("Review strategy - Sharpe ratio below target")
            
        if metrics.consecutive_losses > 3:
            health_check.recommendations.append("Consider reducing position sizes during losing streak")
            
        if emergency_status != EmergencyStatus.NORMAL:
            health_check.recommendations.append("Review and address emergency conditions")
            
        return health_check
        
    def get_risk_dashboard(self) -> Dict:
        """Get comprehensive risk management dashboard data"""
        performance_summary = self.performance_monitor.get_performance_summary()
        limits_summary = self.limits_controller.get_limits_summary()
        market_summary = self.market_analyzer.get_market_risk_summary()
        syndicate_status = self.syndicate_controller.get_syndicate_status()
        
        # Recent decisions summary
        recent_decisions = self.decision_history[-10:]
        approval_rate = (
            sum(1 for d in recent_decisions if d.approved) / len(recent_decisions)
            if recent_decisions else 0
        )
        
        return {
            'framework': {
                'syndicate_name': self.syndicate_name,
                'current_bankroll': self.current_bankroll,
                'active_bets': len(self.active_bets),
                'total_exposure': sum(self.active_bets.values()) if self.active_bets else 0,
                'approval_rate': f"{approval_rate:.1%}"
            },
            'performance': performance_summary,
            'limits': limits_summary,
            'market_risk': market_summary,
            'syndicate': syndicate_status,
            'recent_decisions': [
                {
                    'bet_id': d.bet_id,
                    'approved': d.approved,
                    'amount': d.allocated_amount,
                    'risk_score': d.risk_score,
                    'violations': len(d.violations)
                }
                for d in recent_decisions
            ]
        }
        
    def export_configuration(self) -> Dict:
        """Export current risk management configuration"""
        return {
            'syndicate_name': self.syndicate_name,
            'config': self.config,
            'limits': {
                name: {
                    'type': limit.limit_type.value,
                    'value': limit.value
                }
                for name, limit in self.limits_controller.limits.items()
            },
            'circuit_breakers': [
                {
                    'name': cb.name,
                    'trigger_condition': cb.trigger_condition,
                    'trigger_value': cb.trigger_value,
                    'cooldown_minutes': cb.cooldown_minutes
                }
                for cb in self.limits_controller.circuit_breakers
            ],
            'consensus_requirements': [
                {
                    'min_amount': req.min_amount,
                    'max_amount': req.max_amount,
                    'consensus_type': req.consensus_type.value,
                    'min_participants': req.min_participants
                }
                for req in self.syndicate_controller.consensus_requirements
            ]
        }
        
    def shutdown(self, reason: str):
        """Execute emergency shutdown"""
        logger.critical(f"Risk framework shutdown initiated: {reason}")
        self.syndicate_controller.execute_emergency_shutdown(reason)
        
        # Additional shutdown procedures could be added here
        # - Cancel pending bets
        # - Notify external systems
        # - Save state for recovery