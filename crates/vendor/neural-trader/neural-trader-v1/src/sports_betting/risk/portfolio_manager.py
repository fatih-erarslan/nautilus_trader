"""
Portfolio Risk Manager for Sports Betting

Comprehensive portfolio management with:
- Multi-sport correlation analysis
- Diversification optimization
- Concentration risk monitoring
- VaR and CVaR calculations
- Dynamic position sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
from scipy import stats
from scipy.optimize import minimize
import warnings

from .kelly_criterion import KellyCriterionOptimizer, BettingOpportunity, KellyResult

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Portfolio risk levels"""
    LOW = "low"
    MODERATE = "moderate" 
    HIGH = "high"
    CRITICAL = "critical"


class DiversificationMetric(Enum):
    """Diversification measurement methods"""
    HERFINDAHL = "herfindahl"
    EFFECTIVE_BETS = "effective_bets"
    CORRELATION_ADJUSTED = "correlation_adjusted"
    ENTROPY = "entropy"


@dataclass
class Position:
    """Individual betting position"""
    bet_id: str
    sport: str
    event: str
    selection: str
    stake: float
    odds: float
    probability: float
    entry_time: datetime
    status: str = "active"  # active, settled, cancelled
    result: Optional[str] = None  # win, loss, push, void
    pnl: float = 0.0
    
    def get_exposure(self) -> float:
        """Get current exposure amount"""
        return self.stake if self.status == "active" else 0.0
    
    def get_max_profit(self) -> float:
        """Get maximum possible profit"""
        return self.stake * (self.odds - 1) if self.status == "active" else self.pnl
    
    def get_max_loss(self) -> float:
        """Get maximum possible loss"""
        return -self.stake if self.status == "active" else self.pnl


@dataclass
class Portfolio:
    """Complete betting portfolio"""
    positions: List[Position]
    bankroll: float
    initial_bankroll: float
    creation_time: datetime = field(default_factory=datetime.now)
    
    def get_total_exposure(self) -> float:
        """Get total exposure across all active positions"""
        return sum(pos.get_exposure() for pos in self.positions)
    
    def get_total_max_profit(self) -> float:
        """Get total maximum possible profit"""
        return sum(pos.get_max_profit() for pos in self.positions)
    
    def get_total_max_loss(self) -> float:
        """Get total maximum possible loss"""
        return sum(pos.get_max_loss() for pos in self.positions)
    
    def get_realized_pnl(self) -> float:
        """Get realized P&L from settled positions"""
        return sum(pos.pnl for pos in self.positions if pos.status == "settled")
    
    def get_unrealized_pnl(self) -> float:
        """Get unrealized P&L estimate from active positions"""
        return sum(pos.pnl for pos in self.positions if pos.status == "active")


@dataclass
class RiskMetrics:
    """Portfolio risk metrics"""
    var_1d: float  # 1-day Value at Risk
    cvar_1d: float  # 1-day Conditional VaR
    max_drawdown: float
    concentration_ratio: float
    correlation_risk: float
    leverage_ratio: float
    sharpe_ratio: float
    sortino_ratio: float
    diversification_ratio: float
    risk_level: RiskLevel
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class PortfolioRiskManager:
    """
    Comprehensive portfolio risk management for sports betting operations
    with advanced analytics and real-time monitoring
    """
    
    def __init__(self,
                 initial_bankroll: float,
                 max_exposure_ratio: float = 0.25,
                 max_single_bet_ratio: float = 0.05,
                 max_sport_concentration: float = 0.40,
                 max_event_concentration: float = 0.15,
                 var_confidence: float = 0.05,
                 min_diversification_score: float = 0.6):
        """
        Initialize Portfolio Risk Manager
        
        Args:
            initial_bankroll: Starting bankroll amount
            max_exposure_ratio: Maximum total exposure as % of bankroll
            max_single_bet_ratio: Maximum single bet as % of bankroll
            max_sport_concentration: Maximum concentration in single sport
            max_event_concentration: Maximum concentration in single event
            var_confidence: VaR confidence level (0.05 = 95% VaR)
            min_diversification_score: Minimum diversification score
        """
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.max_exposure_ratio = max_exposure_ratio
        self.max_single_bet_ratio = max_single_bet_ratio
        self.max_sport_concentration = max_sport_concentration
        self.max_event_concentration = max_event_concentration
        self.var_confidence = var_confidence
        self.min_diversification_score = min_diversification_score
        
        # Portfolio tracking
        self.portfolio = Portfolio(
            positions=[],
            bankroll=initial_bankroll,
            initial_bankroll=initial_bankroll
        )
        
        # Risk monitoring
        self.risk_history: List[RiskMetrics] = []
        self.correlation_matrix = pd.DataFrame()
        self.performance_history = []
        
        # Kelly optimizer integration
        self.kelly_optimizer = KellyCriterionOptimizer(
            bankroll=initial_bankroll,
            fractional_factor=0.25
        )
        
        logger.info(f"Portfolio manager initialized with ${initial_bankroll:,.2f} bankroll")
    
    def analyze_portfolio_risk(self, portfolio: Optional[Portfolio] = None) -> RiskMetrics:
        """
        Comprehensive portfolio risk analysis
        
        Args:
            portfolio: Portfolio to analyze (uses current if None)
            
        Returns:
            Complete risk metrics
        """
        if portfolio is None:
            portfolio = self.portfolio
        
        if not portfolio.positions:
            return self._create_empty_risk_metrics()
        
        active_positions = [p for p in portfolio.positions if p.status == "active"]
        
        if not active_positions:
            return self._create_empty_risk_metrics()
        
        # Calculate basic metrics
        total_exposure = portfolio.get_total_exposure()
        exposure_ratio = total_exposure / portfolio.bankroll
        
        # VaR and CVaR calculation
        var_1d, cvar_1d = self._calculate_var_cvar(active_positions)
        
        # Concentration analysis
        concentration_ratio = self._calculate_concentration_ratio(active_positions)
        
        # Correlation risk
        correlation_risk = self._calculate_correlation_risk(active_positions)
        
        # Diversification metrics
        diversification_ratio = self._calculate_diversification_ratio(active_positions)
        
        # Performance metrics
        sharpe_ratio = self._calculate_sharpe_ratio()
        sortino_ratio = self._calculate_sortino_ratio()
        
        # Maximum drawdown
        max_drawdown = self._calculate_max_drawdown()
        
        # Determine risk level
        risk_level = self._determine_risk_level(
            exposure_ratio, concentration_ratio, correlation_risk, diversification_ratio
        )
        
        # Generate warnings and recommendations
        warnings, recommendations = self._generate_risk_alerts(
            exposure_ratio, concentration_ratio, correlation_risk, diversification_ratio
        )
        
        risk_metrics = RiskMetrics(
            var_1d=var_1d,
            cvar_1d=cvar_1d,
            max_drawdown=max_drawdown,
            concentration_ratio=concentration_ratio,
            correlation_risk=correlation_risk,
            leverage_ratio=exposure_ratio,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            diversification_ratio=diversification_ratio,
            risk_level=risk_level,
            warnings=warnings,
            recommendations=recommendations
        )
        
        # Store in history
        self.risk_history.append(risk_metrics)
        
        return risk_metrics
    
    def optimize_portfolio_allocation(self,
                                    opportunities: List[BettingOpportunity],
                                    target_risk_level: RiskLevel = RiskLevel.MODERATE) -> List[KellyResult]:
        """
        Optimize portfolio allocation considering risk constraints
        
        Args:
            opportunities: Available betting opportunities
            target_risk_level: Target portfolio risk level
            
        Returns:
            Optimized allocation results
        """
        # Update Kelly optimizer with current bankroll
        self.kelly_optimizer.update_bankroll(self.current_bankroll)
        
        # Apply risk-level specific constraints
        self._apply_risk_level_constraints(target_risk_level)
        
        # Build correlation matrix for opportunities
        correlation_matrix = self._build_opportunity_correlation_matrix(opportunities)
        
        # Get initial Kelly recommendations
        kelly_results = self.kelly_optimizer.optimize_portfolio(
            opportunities, correlation_matrix
        )
        
        # Apply portfolio-level constraints
        constrained_results = self._apply_portfolio_constraints(kelly_results, opportunities)
        
        # Validate against risk limits
        validated_results = self._validate_risk_limits(constrained_results, opportunities)
        
        return validated_results
    
    def _calculate_var_cvar(self, positions: List[Position]) -> Tuple[float, float]:
        """
        Calculate Value at Risk and Conditional VaR using Monte Carlo simulation
        
        Args:
            positions: Active positions
            
        Returns:
            Tuple of (VaR, CVaR) as positive numbers representing potential losses
        """
        if not positions:
            return 0.0, 0.0
        
        n_simulations = 10000
        portfolio_outcomes = []
        
        for _ in range(n_simulations):
            total_pnl = 0.0
            
            for position in positions:
                # Simulate win/loss based on probability
                if np.random.random() < position.probability:
                    # Win
                    total_pnl += position.stake * (position.odds - 1)
                else:
                    # Loss
                    total_pnl -= position.stake
            
            portfolio_outcomes.append(total_pnl)
        
        portfolio_outcomes = np.array(portfolio_outcomes)
        
        # Calculate VaR (negative of the percentile since we want loss magnitude)
        var = -np.percentile(portfolio_outcomes, self.var_confidence * 100)
        
        # Calculate CVaR (average of losses beyond VaR)
        var_threshold = np.percentile(portfolio_outcomes, self.var_confidence * 100)
        tail_losses = portfolio_outcomes[portfolio_outcomes <= var_threshold]
        cvar = -np.mean(tail_losses) if len(tail_losses) > 0 else var
        
        return var, cvar
    
    def _calculate_concentration_ratio(self, positions: List[Position]) -> float:
        """
        Calculate portfolio concentration using Herfindahl-Hirschman Index
        
        Args:
            positions: Active positions
            
        Returns:
            Concentration ratio (0-1, higher = more concentrated)
        """
        if not positions:
            return 0.0
        
        total_exposure = sum(pos.get_exposure() for pos in positions)
        
        if total_exposure <= 0:
            return 0.0
        
        # Calculate concentration by position size
        position_weights = [pos.get_exposure() / total_exposure for pos in positions]
        hhi = sum(w ** 2 for w in position_weights)
        
        return hhi
    
    def _calculate_correlation_risk(self, positions: List[Position]) -> float:
        """
        Calculate correlation risk based on sport and event overlaps
        
        Args:
            positions: Active positions
            
        Returns:
            Correlation risk score (0-1)
        """
        if len(positions) <= 1:
            return 0.0
        
        total_exposure = sum(pos.get_exposure() for pos in positions)
        
        # Group by sport
        sport_exposure = {}
        for pos in positions:
            sport_exposure[pos.sport] = sport_exposure.get(pos.sport, 0) + pos.get_exposure()
        
        # Calculate sport concentration risk
        sport_weights = [exp / total_exposure for exp in sport_exposure.values()]
        sport_concentration = sum(w ** 2 for w in sport_weights)
        
        # Group by event
        event_exposure = {}
        for pos in positions:
            event_key = f"{pos.sport}_{pos.event}"
            event_exposure[event_key] = event_exposure.get(event_key, 0) + pos.get_exposure()
        
        # Calculate event concentration risk
        event_weights = [exp / total_exposure for exp in event_exposure.values()]
        event_concentration = sum(w ** 2 for w in event_weights)
        
        # Combine sport and event correlation risk
        correlation_risk = 0.6 * sport_concentration + 0.4 * event_concentration
        
        return correlation_risk
    
    def _calculate_diversification_ratio(self, positions: List[Position]) -> float:
        """
        Calculate portfolio diversification ratio
        
        Args:
            positions: Active positions
            
        Returns:
            Diversification ratio (0-1, higher = more diversified)
        """
        if not positions:
            return 0.0
        
        n_positions = len(positions)
        
        # Simple diversification based on number of positions
        position_diversification = min(n_positions / 10, 1.0)  # Normalize to 10 positions
        
        # Sport diversification
        sports = set(pos.sport for pos in positions)
        sport_diversification = min(len(sports) / 5, 1.0)  # Normalize to 5 sports
        
        # Event diversification
        events = set(f"{pos.sport}_{pos.event}" for pos in positions)
        event_diversification = min(len(events) / n_positions, 1.0)
        
        # Combined diversification score
        diversification_ratio = (
            0.4 * position_diversification +
            0.3 * sport_diversification +
            0.3 * event_diversification
        )
        
        return diversification_ratio
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from performance history"""
        if len(self.performance_history) < 2:
            return 0.0
        
        returns = np.diff([p['bankroll'] for p in self.performance_history])
        returns = returns / self.initial_bankroll  # Normalize to initial bankroll
        
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        # Assuming no risk-free rate for simplicity
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        
        return sharpe
    
    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(self.performance_history) < 2:
            return 0.0
        
        returns = np.diff([p['bankroll'] for p in self.performance_history])
        returns = returns / self.initial_bankroll
        
        if len(returns) == 0:
            return 0.0
        
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf')  # No downside
        
        downside_deviation = np.std(negative_returns)
        
        if downside_deviation == 0:
            return 0.0
        
        sortino = np.mean(returns) / downside_deviation * np.sqrt(252)
        
        return sortino
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from performance history"""
        if len(self.performance_history) < 2:
            return 0.0
        
        bankroll_history = [p['bankroll'] for p in self.performance_history]
        peak = bankroll_history[0]
        max_drawdown = 0.0
        
        for bankroll in bankroll_history:
            if bankroll > peak:
                peak = bankroll
            
            drawdown = (peak - bankroll) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _determine_risk_level(self,
                             exposure_ratio: float,
                             concentration_ratio: float,
                             correlation_risk: float,
                             diversification_ratio: float) -> RiskLevel:
        """Determine overall portfolio risk level"""
        risk_score = 0
        
        # Exposure risk
        if exposure_ratio > 0.30:
            risk_score += 3
        elif exposure_ratio > 0.20:
            risk_score += 2
        elif exposure_ratio > 0.10:
            risk_score += 1
        
        # Concentration risk
        if concentration_ratio > 0.50:
            risk_score += 3
        elif concentration_ratio > 0.30:
            risk_score += 2
        elif concentration_ratio > 0.20:
            risk_score += 1
        
        # Correlation risk
        if correlation_risk > 0.70:
            risk_score += 2
        elif correlation_risk > 0.50:
            risk_score += 1
        
        # Diversification penalty
        if diversification_ratio < 0.30:
            risk_score += 2
        elif diversification_ratio < 0.50:
            risk_score += 1
        
        # Map to risk level
        if risk_score >= 7:
            return RiskLevel.CRITICAL
        elif risk_score >= 5:
            return RiskLevel.HIGH
        elif risk_score >= 3:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def _generate_risk_alerts(self,
                             exposure_ratio: float,
                             concentration_ratio: float,
                             correlation_risk: float,
                             diversification_ratio: float) -> Tuple[List[str], List[str]]:
        """Generate risk warnings and recommendations"""
        warnings = []
        recommendations = []
        
        # Exposure warnings
        if exposure_ratio > self.max_exposure_ratio:
            warnings.append(f"Total exposure ({exposure_ratio:.1%}) exceeds limit ({self.max_exposure_ratio:.1%})")
            recommendations.append("Reduce position sizes or close some positions")
        
        # Concentration warnings
        if concentration_ratio > 0.50:
            warnings.append(f"High portfolio concentration (HHI: {concentration_ratio:.3f})")
            recommendations.append("Diversify across more positions")
        
        # Correlation warnings
        if correlation_risk > 0.70:
            warnings.append("High correlation risk - positions may move together")
            recommendations.append("Diversify across different sports and events")
        
        # Diversification warnings
        if diversification_ratio < self.min_diversification_score:
            warnings.append(f"Low diversification score ({diversification_ratio:.2f})")
            recommendations.append("Add positions in different sports/events")
        
        return warnings, recommendations
    
    def _create_empty_risk_metrics(self) -> RiskMetrics:
        """Create empty risk metrics for portfolios with no positions"""
        return RiskMetrics(
            var_1d=0.0,
            cvar_1d=0.0,
            max_drawdown=0.0,
            concentration_ratio=0.0,
            correlation_risk=0.0,
            leverage_ratio=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            diversification_ratio=0.0,
            risk_level=RiskLevel.LOW
        )
    
    def _apply_risk_level_constraints(self, target_risk_level: RiskLevel):
        """Apply risk level specific constraints to Kelly optimizer"""
        risk_multipliers = {
            RiskLevel.LOW: 0.5,
            RiskLevel.MODERATE: 1.0,
            RiskLevel.HIGH: 1.5,
            RiskLevel.CRITICAL: 0.25  # Very conservative for critical
        }
        
        multiplier = risk_multipliers[target_risk_level]
        
        # Adjust Kelly optimizer parameters
        self.kelly_optimizer.fractional_factor *= multiplier
        self.kelly_optimizer.max_allocation = min(
            self.kelly_optimizer.max_allocation,
            self.max_single_bet_ratio * multiplier
        )
    
    def _build_opportunity_correlation_matrix(self, 
                                            opportunities: List[BettingOpportunity]) -> np.ndarray:
        """Build correlation matrix for betting opportunities"""
        n = len(opportunities)
        correlation_matrix = np.eye(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                correlation = 0.0
                
                # Same sport correlation
                if opportunities[i].sport == opportunities[j].sport:
                    correlation += 0.3
                
                # Same event correlation
                if opportunities[i].event == opportunities[j].event:
                    correlation += 0.5
                
                # Cap correlation
                correlation = min(correlation, 0.8)
                
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
        
        return correlation_matrix
    
    def _apply_portfolio_constraints(self,
                                   kelly_results: List[KellyResult],
                                   opportunities: List[BettingOpportunity]) -> List[KellyResult]:
        """Apply portfolio-level constraints to Kelly results"""
        # Calculate current sport exposures
        sport_exposure = {}
        for pos in self.portfolio.positions:
            if pos.status == "active":
                sport_exposure[pos.sport] = sport_exposure.get(pos.sport, 0) + pos.get_exposure()
        
        # Apply constraints
        constrained_results = []
        
        for i, result in enumerate(kelly_results):
            opportunity = opportunities[i]
            
            # Check sport concentration
            current_sport_exposure = sport_exposure.get(opportunity.sport, 0)
            new_exposure = current_sport_exposure + result.recommended_stake
            sport_concentration = new_exposure / self.current_bankroll
            
            if sport_concentration > self.max_sport_concentration:
                # Reduce stake to meet concentration limit
                max_additional = (self.max_sport_concentration * self.current_bankroll) - current_sport_exposure
                result.recommended_stake = max(0, max_additional)
                result.warnings.append("Sport concentration limit applied")
            
            constrained_results.append(result)
        
        return constrained_results
    
    def _validate_risk_limits(self,
                             results: List[KellyResult],
                             opportunities: List[BettingOpportunity]) -> List[KellyResult]:
        """Final validation against all risk limits"""
        total_new_exposure = sum(r.recommended_stake for r in results)
        current_exposure = self.portfolio.get_total_exposure()
        total_exposure = current_exposure + total_new_exposure
        
        # Check total exposure limit
        if total_exposure > self.current_bankroll * self.max_exposure_ratio:
            # Scale down all positions proportionally
            max_new_exposure = (self.current_bankroll * self.max_exposure_ratio) - current_exposure
            if max_new_exposure > 0 and total_new_exposure > 0:
                scale_factor = max_new_exposure / total_new_exposure
                for result in results:
                    result.recommended_stake *= scale_factor
                    result.warnings.append("Total exposure limit scaling applied")
        
        return results
    
    def add_position(self, position: Position):
        """Add new position to portfolio"""
        self.portfolio.positions.append(position)
        self.current_bankroll -= position.stake
        
        # Update performance history
        self.performance_history.append({
            'timestamp': datetime.now(),
            'bankroll': self.current_bankroll,
            'total_exposure': self.portfolio.get_total_exposure(),
            'num_positions': len([p for p in self.portfolio.positions if p.status == "active"])
        })
        
        logger.info(f"Position added: {position.bet_id} - ${position.stake:.2f}")
    
    def close_position(self, bet_id: str, result: str, pnl: float):
        """Close position and update portfolio"""
        for position in self.portfolio.positions:
            if position.bet_id == bet_id and position.status == "active":
                position.status = "settled"
                position.result = result
                position.pnl = pnl
                
                # Update bankroll
                if result == "win":
                    self.current_bankroll += position.stake + pnl
                elif result == "push":
                    self.current_bankroll += position.stake
                # For loss, money already deducted when position opened
                
                # Update Kelly optimizer bankroll
                self.kelly_optimizer.update_bankroll(self.current_bankroll)
                
                # Update performance history
                self.performance_history.append({
                    'timestamp': datetime.now(),
                    'bankroll': self.current_bankroll,
                    'total_exposure': self.portfolio.get_total_exposure(),
                    'num_positions': len([p for p in self.portfolio.positions if p.status == "active"])
                })
                
                logger.info(f"Position closed: {bet_id} - {result} - P&L: ${pnl:.2f}")
                return True
        
        logger.warning(f"Position not found for closing: {bet_id}")
        return False
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        risk_metrics = self.analyze_portfolio_risk()
        
        active_positions = [p for p in self.portfolio.positions if p.status == "active"]
        settled_positions = [p for p in self.portfolio.positions if p.status == "settled"]
        
        return {
            "bankroll": {
                "current": self.current_bankroll,
                "initial": self.initial_bankroll,
                "change": self.current_bankroll - self.initial_bankroll,
                "change_percent": (self.current_bankroll - self.initial_bankroll) / self.initial_bankroll
            },
            "positions": {
                "active": len(active_positions),
                "settled": len(settled_positions),
                "total_exposure": self.portfolio.get_total_exposure(),
                "exposure_ratio": self.portfolio.get_total_exposure() / self.current_bankroll
            },
            "risk_metrics": {
                "var_1d": risk_metrics.var_1d,
                "cvar_1d": risk_metrics.cvar_1d,
                "max_drawdown": risk_metrics.max_drawdown,
                "concentration_ratio": risk_metrics.concentration_ratio,
                "diversification_ratio": risk_metrics.diversification_ratio,
                "risk_level": risk_metrics.risk_level.value
            },
            "performance": {
                "realized_pnl": self.portfolio.get_realized_pnl(),
                "sharpe_ratio": risk_metrics.sharpe_ratio,
                "sortino_ratio": risk_metrics.sortino_ratio
            },
            "alerts": {
                "warnings": risk_metrics.warnings,
                "recommendations": risk_metrics.recommendations
            }
        }


# Example usage
if __name__ == "__main__":
    # Initialize portfolio manager
    manager = PortfolioRiskManager(initial_bankroll=50000)
    
    # Create sample positions
    positions = [
        Position(
            bet_id="nfl_1",
            sport="NFL",
            event="Chiefs vs Bills",
            selection="Chiefs -3.5",
            stake=1000,
            odds=1.91,
            probability=0.55,
            entry_time=datetime.now()
        ),
        Position(
            bet_id="nba_1",
            sport="NBA", 
            event="Lakers vs Warriors",
            selection="Over 215.5",
            stake=750,
            odds=1.85,
            probability=0.60,
            entry_time=datetime.now()
        )
    ]
    
    # Add positions
    for pos in positions:
        manager.add_position(pos)
    
    # Analyze risk
    risk_metrics = manager.analyze_portfolio_risk()
    print(f"Portfolio Risk Level: {risk_metrics.risk_level.value}")
    print(f"VaR (1-day): ${risk_metrics.var_1d:.2f}")
    print(f"Concentration Ratio: {risk_metrics.concentration_ratio:.3f}")
    
    # Get summary
    summary = manager.get_portfolio_summary()
    print(json.dumps(summary, indent=2, default=str))