"""
Integrated Risk Management System for Sports Betting

Unified interface that combines all risk management components:
- Kelly Criterion optimization
- Portfolio risk management
- Circuit breaker protection
- Performance monitoring
- Compliance framework

This serves as the main entry point for all risk management operations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
import uuid

from .kelly_criterion import KellyCriterionOptimizer, BettingOpportunity, KellyResult
from .portfolio_manager import PortfolioRiskManager, Position, RiskLevel
from .circuit_breakers import CircuitBreakerSystem, BreakerTrigger
from .performance_monitor import PerformanceMonitor, BettingTransaction, TimeFrame
from .compliance import ComplianceFramework, Customer, ComplianceStatus

logger = logging.getLogger(__name__)


class SystemStatus(Enum):
    """Overall system status"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical" 
    EMERGENCY_STOP = "emergency_stop"
    MAINTENANCE = "maintenance"


@dataclass
class TradingDecision:
    """Unified trading decision from risk system"""
    decision_id: str
    opportunity: BettingOpportunity
    approved: bool
    recommended_stake: float
    confidence_score: float
    kelly_fraction: float
    risk_score: float
    compliance_status: ComplianceStatus
    warnings: List[str] = field(default_factory=list)
    circuit_breaker_triggered: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SystemHealthReport:
    """Comprehensive system health report"""
    timestamp: datetime
    overall_status: SystemStatus
    bankroll_status: Dict[str, float]
    portfolio_risk: Dict[str, Any]
    circuit_breaker_status: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    compliance_summary: Dict[str, Any]
    active_alerts: List[str]
    recommendations: List[str]


class IntegratedRiskSystem:
    """
    Comprehensive integrated risk management system for sports betting operations.
    
    This system provides a unified interface for all risk management operations,
    coordinating between Kelly optimization, portfolio management, circuit breakers,
    performance monitoring, and compliance checking.
    """
    
    def __init__(self,
                 initial_bankroll: float,
                 syndicate_name: str = "Default Syndicate",
                 risk_tolerance: str = "moderate",  # conservative, moderate, aggressive
                 jurisdiction: str = "Nevada"):
        """
        Initialize Integrated Risk Management System
        
        Args:
            initial_bankroll: Starting bankroll amount
            syndicate_name: Name of the betting syndicate
            risk_tolerance: Risk tolerance level
            jurisdiction: Operating jurisdiction for compliance
        """
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.syndicate_name = syndicate_name
        self.risk_tolerance = risk_tolerance
        self.jurisdiction = jurisdiction
        
        # Initialize all subsystems
        self._initialize_subsystems()
        
        # System state
        self.system_status = SystemStatus.NORMAL
        self.last_health_check = datetime.now()
        self.decision_history: List[TradingDecision] = []
        
        # Integration settings
        self.auto_circuit_breaker_action = True
        self.auto_compliance_check = True
        self.require_unanimous_approval = False
        
        logger.info(f"Integrated risk system initialized for {syndicate_name} with ${initial_bankroll:,.2f}")
    
    def _initialize_subsystems(self):
        """Initialize all risk management subsystems"""
        
        # Kelly Criterion Optimizer
        kelly_params = self._get_kelly_params_by_risk_tolerance()
        self.kelly_optimizer = KellyCriterionOptimizer(
            bankroll=self.initial_bankroll,
            **kelly_params
        )
        
        # Portfolio Risk Manager
        portfolio_params = self._get_portfolio_params_by_risk_tolerance()
        self.portfolio_manager = PortfolioRiskManager(
            initial_bankroll=self.initial_bankroll,
            **portfolio_params
        )
        
        # Circuit Breaker System
        self.circuit_breakers = CircuitBreakerSystem(
            initial_bankroll=self.initial_bankroll
        )
        
        # Performance Monitor
        performance_params = self._get_performance_params_by_risk_tolerance()
        self.performance_monitor = PerformanceMonitor(
            initial_bankroll=self.initial_bankroll,
            **performance_params
        )
        
        # Compliance Framework
        self.compliance = ComplianceFramework()
        
        logger.info("All risk management subsystems initialized")
    
    def _get_kelly_params_by_risk_tolerance(self) -> Dict:
        """Get Kelly optimizer parameters based on risk tolerance"""
        params = {
            "conservative": {
                "fractional_factor": 0.15,
                "max_allocation": 0.03,
                "min_edge": 0.03,
                "confidence_threshold": 0.7
            },
            "moderate": {
                "fractional_factor": 0.25,
                "max_allocation": 0.05,
                "min_edge": 0.02,
                "confidence_threshold": 0.6
            },
            "aggressive": {
                "fractional_factor": 0.40,
                "max_allocation": 0.08,
                "min_edge": 0.015,
                "confidence_threshold": 0.5
            }
        }
        return params.get(self.risk_tolerance, params["moderate"])
    
    def _get_portfolio_params_by_risk_tolerance(self) -> Dict:
        """Get portfolio manager parameters based on risk tolerance"""
        params = {
            "conservative": {
                "max_exposure_ratio": 0.15,
                "max_single_bet_ratio": 0.02,
                "max_sport_concentration": 0.25,
                "min_diversification_score": 0.8
            },
            "moderate": {
                "max_exposure_ratio": 0.25,
                "max_single_bet_ratio": 0.05,
                "max_sport_concentration": 0.40,
                "min_diversification_score": 0.6
            },
            "aggressive": {
                "max_exposure_ratio": 0.40,
                "max_single_bet_ratio": 0.08,
                "max_sport_concentration": 0.60,
                "min_diversification_score": 0.4
            }
        }
        return params.get(self.risk_tolerance, params["moderate"])
    
    def _get_performance_params_by_risk_tolerance(self) -> Dict:
        """Get performance monitor parameters based on risk tolerance"""
        params = {
            "conservative": {
                "max_drawdown_threshold": 0.10,
                "min_sharpe_threshold": 1.5,
                "min_win_rate_threshold": 0.55
            },
            "moderate": {
                "max_drawdown_threshold": 0.15,
                "min_sharpe_threshold": 1.0,
                "min_win_rate_threshold": 0.52
            },
            "aggressive": {
                "max_drawdown_threshold": 0.25,
                "min_sharpe_threshold": 0.8,
                "min_win_rate_threshold": 0.50
            }
        }
        return params.get(self.risk_tolerance, params["moderate"])
    
    def evaluate_betting_opportunity(self,
                                   opportunity: BettingOpportunity,
                                   customer_id: Optional[str] = None,
                                   sport: Optional[str] = None,
                                   bet_type: str = "single") -> TradingDecision:
        """
        Comprehensive evaluation of a betting opportunity through all risk systems
        
        Args:
            opportunity: Betting opportunity to evaluate
            customer_id: Customer ID for compliance check
            sport: Sport type for compliance
            bet_type: Type of bet for compliance
            
        Returns:
            TradingDecision with comprehensive risk assessment
        """
        decision_id = str(uuid.uuid4())
        logger.info(f"Evaluating opportunity: {opportunity.bet_id} (Decision ID: {decision_id})")
        
        # Initialize decision
        decision = TradingDecision(
            decision_id=decision_id,
            opportunity=opportunity,
            approved=False,
            recommended_stake=0.0,
            confidence_score=0.0,
            kelly_fraction=0.0,
            risk_score=0.0,
            compliance_status=ComplianceStatus.PENDING
        )
        
        try:
            # 1. Kelly Criterion Analysis
            kelly_result = self.kelly_optimizer.calculate_single_kelly(opportunity)
            decision.kelly_fraction = kelly_result.kelly_fraction
            decision.recommended_stake = kelly_result.recommended_stake
            decision.warnings.extend(kelly_result.warnings)
            
            if kelly_result.recommended_stake <= 0:
                decision.warnings.append("Kelly optimization recommends no bet")
                return decision
            
            # 2. Portfolio Risk Analysis
            portfolio_context = self._build_portfolio_context()
            portfolio_risk = self.portfolio_manager.analyze_portfolio_risk()
            
            # Check if adding this bet would violate portfolio constraints
            if self._would_violate_portfolio_constraints(kelly_result.recommended_stake, opportunity):
                decision.warnings.append("Portfolio constraints would be violated")
                decision.recommended_stake = self._calculate_max_allowed_stake(opportunity)
            
            # 3. Circuit Breaker Check
            circuit_context = {
                "bankroll": self.current_bankroll,
                "concentration_ratio": portfolio_risk.concentration_ratio
            }
            
            triggered_breakers = self.circuit_breakers.check_all_breakers(circuit_context)
            
            if triggered_breakers:
                decision.circuit_breaker_triggered = True
                for trigger in triggered_breakers:
                    decision.warnings.append(f"Circuit breaker: {trigger.message}")
                
                # Check if any breakers halt betting
                if any(trigger.action.value in ["halt_betting", "emergency_stop"] for trigger in triggered_breakers):
                    decision.warnings.append("Betting halted by circuit breakers")
                    return decision
            
            # 4. Performance Monitoring Check
            performance_alerts = self.performance_monitor.check_performance_alerts()
            for alert in performance_alerts:
                if alert.severity in ["high", "critical"]:
                    decision.warnings.append(f"Performance alert: {alert.message}")
            
            # 5. Compliance Check
            if customer_id and self.auto_compliance_check:
                compliance_status, compliance_violations = self.compliance.check_transaction_compliance(
                    customer_id=customer_id,
                    bet_amount=decision.recommended_stake,
                    sport=sport or opportunity.sport,
                    bet_type=bet_type,
                    jurisdiction=self.jurisdiction
                )
                
                decision.compliance_status = compliance_status
                
                if compliance_violations:
                    decision.warnings.extend([f"Compliance: {v}" for v in compliance_violations])
                    
                    if compliance_status == ComplianceStatus.VIOLATION:
                        decision.warnings.append("Compliance violations prevent betting")
                        return decision
            else:
                decision.compliance_status = ComplianceStatus.COMPLIANT
            
            # 6. Calculate Overall Risk Score and Confidence
            decision.risk_score = self._calculate_overall_risk_score(
                kelly_result, portfolio_risk, triggered_breakers, performance_alerts
            )
            
            decision.confidence_score = self._calculate_confidence_score(
                opportunity, kelly_result, portfolio_risk, decision.warnings
            )
            
            # 7. Make Final Decision
            decision.approved = self._make_final_decision(decision)
            
            # Store metadata
            decision.metadata = {
                "kelly_expected_growth": kelly_result.expected_growth,
                "portfolio_risk_level": portfolio_risk.risk_level.value,
                "circuit_breakers_triggered": len(triggered_breakers),
                "performance_alerts": len(performance_alerts),
                "system_status": self.system_status.value
            }
            
        except Exception as e:
            logger.error(f"Error evaluating opportunity {opportunity.bet_id}: {str(e)}")
            decision.warnings.append(f"System error: {str(e)}")
            decision.approved = False
        
        # Store decision in history
        self.decision_history.append(decision)
        
        logger.info(
            f"Decision complete: {decision_id} - "
            f"{'APPROVED' if decision.approved else 'REJECTED'} - "
            f"Stake: ${decision.recommended_stake:.2f} - "
            f"Risk: {decision.risk_score:.2f} - "
            f"Confidence: {decision.confidence_score:.2f}"
        )
        
        return decision
    
    def execute_approved_bet(self, decision: TradingDecision) -> bool:
        """
        Execute an approved betting decision
        
        Args:
            decision: Approved trading decision
            
        Returns:
            True if bet executed successfully
        """
        if not decision.approved:
            logger.error(f"Attempting to execute unapproved decision: {decision.decision_id}")
            return False
        
        if decision.recommended_stake <= 0:
            logger.error(f"Invalid stake amount: ${decision.recommended_stake}")
            return False
        
        try:
            # Create position
            position = Position(
                bet_id=decision.opportunity.bet_id,
                sport=decision.opportunity.sport,
                event=decision.opportunity.event,
                selection=decision.opportunity.selection,
                stake=decision.recommended_stake,
                odds=decision.opportunity.odds,
                probability=decision.opportunity.probability,
                entry_time=datetime.now()
            )
            
            # Add to portfolio
            self.portfolio_manager.add_position(position)
            
            # Create transaction for performance monitoring
            transaction = BettingTransaction(
                transaction_id=decision.opportunity.bet_id,
                timestamp=datetime.now(),
                sport=decision.opportunity.sport,
                event=decision.opportunity.event,
                selection=decision.opportunity.selection,
                bet_type="single",  # Could be enhanced
                stake=decision.recommended_stake,
                odds=decision.opportunity.odds
            )
            
            self.performance_monitor.record_transaction(transaction)
            
            # Update bankroll
            self.current_bankroll -= decision.recommended_stake
            self._update_all_bankrolls()
            
            logger.info(f"Bet executed: {decision.opportunity.bet_id} - ${decision.recommended_stake:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing bet {decision.opportunity.bet_id}: {str(e)}")
            return False
    
    def settle_bet(self, bet_id: str, result: str, pnl: float) -> bool:
        """
        Settle a completed bet
        
        Args:
            bet_id: Bet identifier
            result: Bet result (win, loss, push, void)
            pnl: Profit/loss amount
            
        Returns:
            True if settled successfully
        """
        try:
            # Update portfolio
            portfolio_updated = self.portfolio_manager.close_position(bet_id, result, pnl)
            
            # Update performance monitor
            performance_updated = self.performance_monitor.update_transaction_result(bet_id, result, pnl)
            
            # Update circuit breakers
            self.circuit_breakers.update_bankroll(self.current_bankroll, result)
            
            if portfolio_updated and performance_updated:
                logger.info(f"Bet settled: {bet_id} - {result} - P&L: ${pnl:.2f}")
                
                # Run health check after settlement
                self.perform_health_check()
                
                return True
            else:
                logger.error(f"Failed to settle bet: {bet_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error settling bet {bet_id}: {str(e)}")
            return False
    
    def perform_health_check(self) -> SystemHealthReport:
        """
        Comprehensive system health check
        
        Returns:
            SystemHealthReport with all system status information
        """
        logger.info("Performing comprehensive system health check")
        
        try:
            # Get current metrics from all subsystems
            portfolio_risk = self.portfolio_manager.analyze_portfolio_risk()
            circuit_status = self.circuit_breakers.get_system_status()
            performance_metrics = self.performance_monitor.get_performance_summary()
            compliance_dashboard = self.compliance.get_compliance_dashboard()
            
            # Collect active alerts
            active_alerts = []
            
            # Portfolio warnings
            active_alerts.extend(portfolio_risk.warnings)
            
            # Circuit breaker issues
            if circuit_status["system_status"]["halted"]:
                active_alerts.append("Circuit breaker system halted")
            if circuit_status["system_status"]["emergency_stop"]:
                active_alerts.append("Emergency stop activated")
            
            # Performance alerts
            perf_alerts = self.performance_monitor.check_performance_alerts()
            active_alerts.extend([alert.message for alert in perf_alerts if not alert.resolved])
            
            # Compliance events
            compliance_events = [e for e in self.compliance.compliance_events 
                               if e.timestamp > datetime.now() - timedelta(days=1) and not e.auto_resolved]
            active_alerts.extend([f"Compliance: {event.description}" for event in compliance_events])
            
            # Determine overall system status
            overall_status = self._determine_overall_system_status(
                portfolio_risk, circuit_status, performance_metrics, compliance_events
            )
            
            # Generate recommendations
            recommendations = self._generate_system_recommendations(
                portfolio_risk, circuit_status, performance_metrics, compliance_dashboard
            )
            
            # Create health report
            health_report = SystemHealthReport(
                timestamp=datetime.now(),
                overall_status=overall_status,
                bankroll_status={
                    "current": self.current_bankroll,
                    "initial": self.initial_bankroll,
                    "peak": getattr(self.portfolio_manager.portfolio, 'peak_bankroll', self.initial_bankroll),
                    "change_percent": (self.current_bankroll - self.initial_bankroll) / self.initial_bankroll
                },
                portfolio_risk={
                    "risk_level": portfolio_risk.risk_level.value,
                    "var_1d": portfolio_risk.var_1d,
                    "max_drawdown": portfolio_risk.max_drawdown,
                    "concentration_ratio": portfolio_risk.concentration_ratio,
                    "diversification_ratio": portfolio_risk.diversification_ratio
                },
                circuit_breaker_status=circuit_status,
                performance_metrics=performance_metrics,
                compliance_summary=compliance_dashboard,
                active_alerts=active_alerts,
                recommendations=recommendations
            )
            
            self.system_status = overall_status
            self.last_health_check = datetime.now()
            
            logger.info(f"Health check complete - Status: {overall_status.value}")
            
            return health_report
            
        except Exception as e:
            logger.error(f"Error during health check: {str(e)}")
            
            # Return emergency status
            return SystemHealthReport(
                timestamp=datetime.now(),
                overall_status=SystemStatus.CRITICAL,
                bankroll_status={"current": self.current_bankroll},
                portfolio_risk={},
                circuit_breaker_status={},
                performance_metrics={},
                compliance_summary={},
                active_alerts=[f"Health check error: {str(e)}"],
                recommendations=["Immediate system review required"]
            )
    
    def _build_portfolio_context(self) -> Dict[str, Any]:
        """Build context for portfolio analysis"""
        return {
            "current_bankroll": self.current_bankroll,
            "jurisdiction": self.jurisdiction,
            "risk_tolerance": self.risk_tolerance
        }
    
    def _would_violate_portfolio_constraints(self, stake: float, opportunity: BettingOpportunity) -> bool:
        """Check if bet would violate portfolio constraints"""
        current_exposure = self.portfolio_manager.portfolio.get_total_exposure()
        new_total_exposure = current_exposure + stake
        
        max_allowed_exposure = self.current_bankroll * self.portfolio_manager.max_exposure_ratio
        
        return new_total_exposure > max_allowed_exposure
    
    def _calculate_max_allowed_stake(self, opportunity: BettingOpportunity) -> float:
        """Calculate maximum allowed stake given constraints"""
        current_exposure = self.portfolio_manager.portfolio.get_total_exposure()
        max_allowed_exposure = self.current_bankroll * self.portfolio_manager.max_exposure_ratio
        max_additional = max_allowed_exposure - current_exposure
        
        # Also check single bet limit
        max_single_bet = self.current_bankroll * self.portfolio_manager.max_single_bet_ratio
        
        return max(0, min(max_additional, max_single_bet))
    
    def _calculate_overall_risk_score(self,
                                    kelly_result: KellyResult,
                                    portfolio_risk: Any,
                                    triggered_breakers: List[BreakerTrigger],
                                    performance_alerts: List[Any]) -> float:
        """Calculate overall risk score (0-1)"""
        risk_components = {
            "kelly_risk": min(kelly_result.kelly_fraction / 0.5, 1.0),  # Normalize to 50% Kelly
            "portfolio_risk": {
                RiskLevel.LOW: 0.2,
                RiskLevel.MODERATE: 0.5,
                RiskLevel.HIGH: 0.8,
                RiskLevel.CRITICAL: 1.0
            }.get(portfolio_risk.risk_level, 0.5),
            "circuit_breaker_risk": len(triggered_breakers) * 0.2,
            "performance_risk": len([a for a in performance_alerts if a.severity in ["high", "critical"]]) * 0.15
        }
        
        # Weighted average
        weights = [0.3, 0.4, 0.2, 0.1]
        risk_score = sum(w * r for w, r in zip(weights, risk_components.values()))
        
        return min(risk_score, 1.0)
    
    def _calculate_confidence_score(self,
                                  opportunity: BettingOpportunity,
                                  kelly_result: KellyResult,
                                  portfolio_risk: Any,
                                  warnings: List[str]) -> float:
        """Calculate confidence score for the decision (0-1)"""
        confidence_factors = {
            "opportunity_confidence": opportunity.confidence,
            "edge_strength": min(opportunity.edge / 0.1, 1.0),  # Normalize to 10% edge
            "kelly_growth": min(kelly_result.expected_growth / 0.1, 1.0),
            "portfolio_stability": 1.0 - portfolio_risk.concentration_ratio,
            "warning_penalty": max(0, 1.0 - len(warnings) * 0.1)
        }
        
        # Weighted average
        weights = [0.25, 0.25, 0.2, 0.15, 0.15]
        confidence = sum(w * f for w, f in zip(weights, confidence_factors.values()))
        
        return max(0, min(confidence, 1.0))
    
    def _make_final_decision(self, decision: TradingDecision) -> bool:
        """Make final approval decision based on all factors"""
        # Must have positive stake
        if decision.recommended_stake <= 0:
            return False
        
        # Check compliance
        if decision.compliance_status == ComplianceStatus.VIOLATION:
            return False
        
        # Check circuit breakers
        if decision.circuit_breaker_triggered:
            # Only approve if no halt/emergency actions
            critical_warnings = [w for w in decision.warnings if "halt" in w.lower() or "emergency" in w.lower()]
            if critical_warnings:
                return False
        
        # Risk and confidence thresholds
        risk_threshold = {"conservative": 0.6, "moderate": 0.7, "aggressive": 0.8}[self.risk_tolerance]
        confidence_threshold = {"conservative": 0.7, "moderate": 0.6, "aggressive": 0.5}[self.risk_tolerance]
        
        if decision.risk_score > risk_threshold:
            return False
        
        if decision.confidence_score < confidence_threshold:
            return False
        
        return True
    
    def _update_all_bankrolls(self):
        """Update bankroll across all subsystems"""
        self.kelly_optimizer.update_bankroll(self.current_bankroll)
        # Portfolio manager bankroll is updated when positions are added/closed
        # Circuit breakers are updated when bets are settled
        # Performance monitor tracks bankroll through transactions
    
    def _determine_overall_system_status(self,
                                       portfolio_risk: Any,
                                       circuit_status: Dict,
                                       performance_metrics: Dict,
                                       compliance_events: List) -> SystemStatus:
        """Determine overall system status"""
        
        # Emergency conditions
        if circuit_status["system_status"]["emergency_stop"]:
            return SystemStatus.EMERGENCY_STOP
        
        # Critical conditions
        critical_conditions = [
            portfolio_risk.risk_level == RiskLevel.CRITICAL,
            circuit_status["system_status"]["halted"],
            len([e for e in compliance_events if e.severity == "critical"]) > 0,
            self.current_bankroll < self.initial_bankroll * 0.5  # 50% loss
        ]
        
        if any(critical_conditions):
            return SystemStatus.CRITICAL
        
        # Warning conditions
        warning_conditions = [
            portfolio_risk.risk_level == RiskLevel.HIGH,
            len(circuit_status["recent_triggers"]) > 3,
            performance_metrics.get("risk", {}).get("current_drawdown", 0) > 0.1,
            len([e for e in compliance_events if e.severity == "high"]) > 2
        ]
        
        if any(warning_conditions):
            return SystemStatus.WARNING
        
        return SystemStatus.NORMAL
    
    def _generate_system_recommendations(self,
                                       portfolio_risk: Any,
                                       circuit_status: Dict,
                                       performance_metrics: Dict,
                                       compliance_dashboard: Dict) -> List[str]:
        """Generate system recommendations"""
        recommendations = []
        
        # Portfolio recommendations
        recommendations.extend(portfolio_risk.recommendations)
        
        # Circuit breaker recommendations
        if circuit_status["system_status"]["halted"]:
            recommendations.append("Review and resolve circuit breaker triggers before resuming")
        
        # Performance recommendations
        current_dd = performance_metrics.get("risk", {}).get("current_drawdown", 0)
        if current_dd > 0.1:
            recommendations.append("Review strategy performance - significant drawdown detected")
        
        sharpe = performance_metrics.get("risk", {}).get("sharpe_ratio", 0)
        if sharpe < 1.0:
            recommendations.append("Review strategy effectiveness - Sharpe ratio below target")
        
        # Compliance recommendations
        pending_kyc = compliance_dashboard.get("customers", {}).get("kyc_summary", {}).get("pending", 0)
        if pending_kyc > 0:
            recommendations.append(f"Complete KYC for {pending_kyc} pending customers")
        
        return recommendations
    
    def get_system_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive system dashboard"""
        health_report = self.perform_health_check()
        
        # Recent decisions summary
        recent_decisions = self.decision_history[-10:]
        approval_rate = sum(1 for d in recent_decisions if d.approved) / len(recent_decisions) if recent_decisions else 0
        
        return {
            "system_overview": {
                "status": health_report.overall_status.value,
                "syndicate_name": self.syndicate_name,
                "risk_tolerance": self.risk_tolerance,
                "jurisdiction": self.jurisdiction,
                "last_health_check": health_report.timestamp.isoformat()
            },
            "bankroll": health_report.bankroll_status,
            "portfolio": health_report.portfolio_risk,
            "circuit_breakers": health_report.circuit_breaker_status,
            "performance": health_report.performance_metrics,
            "compliance": health_report.compliance_summary,
            "decisions": {
                "total": len(self.decision_history),
                "recent_approval_rate": f"{approval_rate:.1%}",
                "avg_stake": np.mean([d.recommended_stake for d in recent_decisions]) if recent_decisions else 0,
                "avg_confidence": np.mean([d.confidence_score for d in recent_decisions]) if recent_decisions else 0
            },
            "alerts": {
                "count": len(health_report.active_alerts),
                "alerts": health_report.active_alerts[:5],  # Top 5 alerts
                "recommendations": health_report.recommendations
            }
        }


# Example usage
if __name__ == "__main__":
    # Initialize integrated risk system
    risk_system = IntegratedRiskSystem(
        initial_bankroll=100000,
        syndicate_name="Test Syndicate",
        risk_tolerance="moderate",
        jurisdiction="Nevada"
    )
    
    # Create sample betting opportunity
    opportunity = BettingOpportunity(
        bet_id="nfl_001",
        odds=1.91,
        probability=0.55,
        confidence=0.8,
        sport="NFL",
        event="Chiefs vs Bills",
        selection="Chiefs -3.5"
    )
    
    # Evaluate opportunity
    decision = risk_system.evaluate_betting_opportunity(opportunity)
    
    print(f"Decision: {'APPROVED' if decision.approved else 'REJECTED'}")
    print(f"Recommended Stake: ${decision.recommended_stake:.2f}")
    print(f"Risk Score: {decision.risk_score:.2f}")
    print(f"Confidence: {decision.confidence_score:.2f}")
    
    if decision.warnings:
        print("Warnings:")
        for warning in decision.warnings:
            print(f"  - {warning}")
    
    # Get system dashboard
    dashboard = risk_system.get_system_dashboard()
    print(json.dumps(dashboard, indent=2, default=str))