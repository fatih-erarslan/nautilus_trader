"""
Kelly Criterion Optimization for Sports Betting

Advanced implementation of Kelly Criterion with:
- Fractional Kelly adjustments
- Edge calculation and probability assessment
- Portfolio-level optimization
- Risk-adjusted position sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy.optimize import minimize
import warnings

logger = logging.getLogger(__name__)


class KellyMethod(Enum):
    """Kelly calculation methods"""
    SIMPLE = "simple"
    FRACTIONAL = "fractional" 
    SIMULTANEOUS = "simultaneous"
    CONSTRAINED = "constrained"


@dataclass
class BettingOpportunity:
    """Represents a single betting opportunity"""
    bet_id: str
    odds: float  # Decimal odds
    probability: float  # True probability estimate
    confidence: float = 1.0  # Confidence in probability (0-1)
    correlation_factor: float = 0.0  # Correlation with other bets
    max_stake: Optional[float] = None
    sport: str = ""
    event: str = ""
    selection: str = ""
    
    def __post_init__(self):
        """Validate and calculate derived metrics"""
        if self.odds <= 1.0:
            raise ValueError("Odds must be greater than 1.0")
        if not 0 < self.probability < 1:
            raise ValueError("Probability must be between 0 and 1")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
            
        # Calculate edge
        self.edge = self.probability - (1 / self.odds)
        self.implied_probability = 1 / self.odds
        self.profit_ratio = self.odds - 1
        

@dataclass
class KellyResult:
    """Result of Kelly criterion calculation"""
    bet_id: str
    kelly_fraction: float
    fractional_kelly: float
    recommended_stake: float
    expected_growth: float
    edge: float
    variance: float
    confidence_adjusted: bool
    warnings: List[str] = field(default_factory=list)


class KellyCriterionOptimizer:
    """
    Advanced Kelly Criterion optimizer for sports betting with
    portfolio-level optimization and risk management
    """
    
    def __init__(self,
                 bankroll: float,
                 fractional_factor: float = 0.25,
                 max_allocation: float = 0.10,
                 min_edge: float = 0.02,
                 max_correlation: float = 0.7,
                 confidence_threshold: float = 0.5):
        """
        Initialize Kelly Criterion Optimizer
        
        Args:
            bankroll: Available bankroll
            fractional_factor: Fraction of Kelly to use (0.25 = 25% of Kelly)
            max_allocation: Maximum allocation per bet as fraction of bankroll
            min_edge: Minimum edge required to consider a bet
            max_correlation: Maximum allowed correlation between bets
            confidence_threshold: Minimum confidence required for full allocation
        """
        self.bankroll = bankroll
        self.fractional_factor = fractional_factor
        self.max_allocation = max_allocation
        self.min_edge = min_edge
        self.max_correlation = max_correlation
        self.confidence_threshold = confidence_threshold
        
        # Risk parameters
        self.risk_aversion = 1.0  # Risk aversion parameter
        self.max_drawdown_prob = 0.01  # Max probability of 50%+ drawdown
        
        logger.info(f"Kelly optimizer initialized with bankroll: ${bankroll:,.2f}")
    
    def calculate_single_kelly(self, opportunity: BettingOpportunity) -> KellyResult:
        """
        Calculate Kelly criterion for a single betting opportunity
        
        Args:
            opportunity: Betting opportunity to analyze
            
        Returns:
            KellyResult with recommended stake and metrics
        """
        if opportunity.edge < self.min_edge:
            logger.warning(f"Bet {opportunity.bet_id} has insufficient edge: {opportunity.edge:.4f}")
            return KellyResult(
                bet_id=opportunity.bet_id,
                kelly_fraction=0.0,
                fractional_kelly=0.0,
                recommended_stake=0.0,
                expected_growth=0.0,
                edge=opportunity.edge,
                variance=0.0,
                confidence_adjusted=False,
                warnings=["Insufficient edge"]
            )
        
        # Standard Kelly formula: f* = (bp - q) / b
        # where p = win probability, q = loss probability, b = profit ratio
        p = opportunity.probability
        q = 1 - p
        b = opportunity.profit_ratio
        
        kelly_fraction = (b * p - q) / b
        
        # Apply confidence adjustment
        confidence_adjusted = opportunity.confidence < self.confidence_threshold
        if confidence_adjusted:
            kelly_fraction *= opportunity.confidence
            
        # Calculate variance for risk assessment
        variance = p * (b ** 2) + q * (-1) ** 2 - (p * b - q) ** 2
        
        # Apply fractional Kelly
        fractional_kelly = kelly_fraction * self.fractional_factor
        
        # Apply maximum allocation constraint
        final_fraction = min(fractional_kelly, self.max_allocation)
        
        # Calculate recommended stake
        recommended_stake = self.bankroll * final_fraction
        
        # Apply maximum stake constraint if specified
        if opportunity.max_stake is not None:
            recommended_stake = min(recommended_stake, opportunity.max_stake)
            final_fraction = recommended_stake / self.bankroll
            
        # Calculate expected growth rate
        expected_growth = self._calculate_expected_growth(
            final_fraction, opportunity.probability, opportunity.profit_ratio
        )
        
        # Generate warnings
        warnings = []
        if kelly_fraction > 0.5:
            warnings.append("High Kelly fraction - consider reducing")
        if kelly_fraction != fractional_kelly:
            warnings.append("Fractional Kelly applied")
        if fractional_kelly != final_fraction:
            warnings.append("Maximum allocation constraint applied")
        if confidence_adjusted:
            warnings.append("Confidence adjustment applied")
            
        return KellyResult(
            bet_id=opportunity.bet_id,
            kelly_fraction=kelly_fraction,
            fractional_kelly=fractional_kelly,
            recommended_stake=recommended_stake,
            expected_growth=expected_growth,
            edge=opportunity.edge,
            variance=variance,
            confidence_adjusted=confidence_adjusted,
            warnings=warnings
        )
    
    def optimize_portfolio(self, 
                          opportunities: List[BettingOpportunity],
                          correlation_matrix: Optional[np.ndarray] = None) -> List[KellyResult]:
        """
        Optimize portfolio allocation across multiple betting opportunities
        
        Args:
            opportunities: List of betting opportunities
            correlation_matrix: Optional correlation matrix between bets
            
        Returns:
            List of optimized Kelly results
        """
        n_bets = len(opportunities)
        if n_bets == 0:
            return []
            
        # Filter by minimum edge
        valid_opportunities = [opp for opp in opportunities if opp.edge >= self.min_edge]
        
        if not valid_opportunities:
            logger.warning("No opportunities meet minimum edge requirement")
            return [self.calculate_single_kelly(opp) for opp in opportunities]
        
        # If only one opportunity, use single Kelly
        if len(valid_opportunities) == 1:
            return [self.calculate_single_kelly(valid_opportunities[0])]
        
        # Build correlation matrix if not provided
        if correlation_matrix is None:
            correlation_matrix = self._build_correlation_matrix(valid_opportunities)
        
        # Calculate simultaneous Kelly fractions
        simultaneous_results = self._calculate_simultaneous_kelly(
            valid_opportunities, correlation_matrix
        )
        
        # Apply constraints and create results
        results = []
        for i, opp in enumerate(opportunities):
            if opp in valid_opportunities:
                idx = valid_opportunities.index(opp)
                result = simultaneous_results[idx]
            else:
                # Zero allocation for invalid opportunities
                result = KellyResult(
                    bet_id=opp.bet_id,
                    kelly_fraction=0.0,
                    fractional_kelly=0.0,
                    recommended_stake=0.0,
                    expected_growth=0.0,
                    edge=opp.edge,
                    variance=0.0,
                    confidence_adjusted=False,
                    warnings=["Insufficient edge"]
                )
            results.append(result)
        
        return results
    
    def _calculate_simultaneous_kelly(self,
                                     opportunities: List[BettingOpportunity],
                                     correlation_matrix: np.ndarray) -> List[KellyResult]:
        """
        Calculate Kelly fractions for multiple correlated bets simultaneously
        
        Args:
            opportunities: Valid betting opportunities
            correlation_matrix: Correlation matrix
            
        Returns:
            List of Kelly results
        """
        n = len(opportunities)
        
        # Build expected returns vector
        expected_returns = np.array([opp.edge for opp in opportunities])
        
        # Build covariance matrix
        variances = np.array([
            opp.probability * (opp.profit_ratio ** 2) + (1 - opp.probability) * 1 
            - (opp.probability * opp.profit_ratio - (1 - opp.probability)) ** 2
            for opp in opportunities
        ])
        
        covariance_matrix = np.outer(np.sqrt(variances), np.sqrt(variances)) * correlation_matrix
        
        # Solve for optimal fractions using quadratic programming
        try:
            # Use mean-variance optimization with Kelly-inspired objective
            fractions = self._solve_portfolio_optimization(
                expected_returns, covariance_matrix, opportunities
            )
        except Exception as e:
            logger.warning(f"Portfolio optimization failed: {e}, falling back to individual Kelly")
            fractions = np.array([
                self.calculate_single_kelly(opp).fractional_kelly 
                for opp in opportunities
            ])
        
        # Create results
        results = []
        for i, opp in enumerate(opportunities):
            fraction = fractions[i]
            stake = self.bankroll * fraction
            
            # Calculate expected growth
            expected_growth = self._calculate_expected_growth(
                fraction, opp.probability, opp.profit_ratio
            )
            
            # Generate warnings
            warnings = []
            single_kelly = self.calculate_single_kelly(opp).kelly_fraction
            if abs(fraction - single_kelly * self.fractional_factor) > 0.01:
                warnings.append("Portfolio optimization adjustment applied")
            
            result = KellyResult(
                bet_id=opp.bet_id,
                kelly_fraction=single_kelly,
                fractional_kelly=fraction,
                recommended_stake=stake,
                expected_growth=expected_growth,
                edge=opp.edge,
                variance=variances[i],
                confidence_adjusted=opp.confidence < self.confidence_threshold,
                warnings=warnings
            )
            results.append(result)
        
        return results
    
    def _solve_portfolio_optimization(self,
                                     expected_returns: np.ndarray,
                                     covariance_matrix: np.ndarray,
                                     opportunities: List[BettingOpportunity]) -> np.ndarray:
        """
        Solve portfolio optimization problem
        
        Args:
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            opportunities: Betting opportunities
            
        Returns:
            Optimal fraction allocations
        """
        n = len(expected_returns)
        
        # Initial guess - fractional Kelly for each bet
        x0 = np.array([
            self.calculate_single_kelly(opp).fractional_kelly 
            for opp in opportunities
        ])
        
        # Objective function - maximize Kelly growth with risk penalty
        def objective(fractions):
            portfolio_return = np.dot(fractions, expected_returns)
            portfolio_variance = np.dot(fractions, np.dot(covariance_matrix, fractions))
            
            # Kelly-inspired utility with risk aversion
            utility = portfolio_return - 0.5 * self.risk_aversion * portfolio_variance
            return -utility  # Minimize negative utility
        
        # Constraints
        constraints = []
        
        # Total allocation constraint
        constraints.append({
            'type': 'ineq',
            'fun': lambda x: self.max_allocation * n - np.sum(x)  # Sum <= max_allocation * n
        })
        
        # Individual allocation constraints
        bounds = [(0, self.max_allocation) for _ in range(n)]
        
        # Solve optimization
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            return x0  # Fall back to initial guess
        
        return result.x
    
    def _build_correlation_matrix(self, 
                                 opportunities: List[BettingOpportunity]) -> np.ndarray:
        """
        Build correlation matrix based on opportunity characteristics
        
        Args:
            opportunities: List of betting opportunities
            
        Returns:
            Correlation matrix
        """
        n = len(opportunities)
        correlation_matrix = np.eye(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                correlation = 0.0
                
                # Same sport correlation
                if opportunities[i].sport == opportunities[j].sport:
                    correlation += 0.2
                
                # Same event correlation
                if opportunities[i].event == opportunities[j].event:
                    correlation += 0.5
                
                # Add manual correlation factors
                correlation += max(
                    opportunities[i].correlation_factor,
                    opportunities[j].correlation_factor
                )
                
                # Cap correlation
                correlation = min(correlation, self.max_correlation)
                
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
        
        return correlation_matrix
    
    def _calculate_expected_growth(self,
                                  fraction: float,
                                  win_prob: float,
                                  profit_ratio: float) -> float:
        """
        Calculate expected logarithmic growth rate
        
        Args:
            fraction: Fraction of bankroll bet
            win_prob: Probability of winning
            profit_ratio: Profit ratio (odds - 1)
            
        Returns:
            Expected growth rate
        """
        if fraction <= 0:
            return 0.0
        
        # Expected log growth
        win_growth = np.log(1 + fraction * profit_ratio)
        loss_growth = np.log(1 - fraction)
        
        expected_growth = win_prob * win_growth + (1 - win_prob) * loss_growth
        
        return expected_growth
    
    def calculate_drawdown_probability(self,
                                      results: List[KellyResult],
                                      num_simulations: int = 10000,
                                      time_horizon: int = 100) -> Dict[str, float]:
        """
        Calculate probability of various drawdown levels using Monte Carlo
        
        Args:
            results: List of Kelly results
            num_simulations: Number of Monte Carlo simulations
            time_horizon: Number of betting periods to simulate
            
        Returns:
            Dictionary of drawdown probabilities
        """
        if not results:
            return {}
        
        # Extract parameters
        fractions = np.array([r.fractional_kelly for r in results])
        edges = np.array([r.edge for r in results])
        total_fraction = np.sum(fractions)
        
        if total_fraction <= 0:
            return {"max_drawdown": 0.0, "prob_10_percent": 0.0, "prob_20_percent": 0.0}
        
        # Simulate bankroll evolution
        max_drawdowns = []
        
        for _ in range(num_simulations):
            bankroll = 1.0  # Start with normalized bankroll
            peak = 1.0
            max_drawdown = 0.0
            
            for t in range(time_horizon):
                # Simulate bet outcomes
                total_return = 0.0
                
                for i, result in enumerate(results):
                    if result.fractional_kelly > 0:
                        # Simulate win/loss
                        if np.random.random() < result.edge + (1 / (result.edge * 10 + 1)):  # Approximate win prob
                            # Win
                            profit_ratio = 1.0 / result.edge - 1  # Approximate odds
                            total_return += result.fractional_kelly * profit_ratio
                        else:
                            # Loss
                            total_return -= result.fractional_kelly
                
                # Update bankroll
                bankroll *= (1 + total_return)
                
                # Track drawdown
                if bankroll > peak:
                    peak = bankroll
                
                current_drawdown = (peak - bankroll) / peak
                max_drawdown = max(max_drawdown, current_drawdown)
            
            max_drawdowns.append(max_drawdown)
        
        # Calculate statistics
        max_drawdowns = np.array(max_drawdowns)
        
        return {
            "expected_max_drawdown": np.mean(max_drawdowns),
            "prob_10_percent": np.mean(max_drawdowns >= 0.10),
            "prob_20_percent": np.mean(max_drawdowns >= 0.20),
            "prob_50_percent": np.mean(max_drawdowns >= 0.50),
            "percentile_95": np.percentile(max_drawdowns, 95)
        }
    
    def get_portfolio_summary(self, results: List[KellyResult]) -> Dict[str, float]:
        """
        Get portfolio-level summary statistics
        
        Args:
            results: List of Kelly results
            
        Returns:
            Portfolio summary metrics
        """
        if not results:
            return {}
        
        total_allocation = sum(r.recommended_stake for r in results)
        total_fraction = total_allocation / self.bankroll
        
        valid_results = [r for r in results if r.recommended_stake > 0]
        
        portfolio_edge = sum(r.edge * r.fractional_kelly for r in valid_results)
        expected_growth = sum(r.expected_growth * r.fractional_kelly for r in valid_results)
        
        return {
            "total_allocation": total_allocation,
            "total_fraction": total_fraction,
            "num_bets": len(valid_results),
            "portfolio_edge": portfolio_edge,
            "expected_growth": expected_growth,
            "max_single_allocation": max((r.recommended_stake for r in results), default=0),
            "avg_edge": np.mean([r.edge for r in valid_results]) if valid_results else 0,
            "diversification_ratio": len(valid_results) / len(results) if results else 0
        }
    
    def update_bankroll(self, new_bankroll: float):
        """Update bankroll and log the change"""
        old_bankroll = self.bankroll
        self.bankroll = new_bankroll
        change = new_bankroll - old_bankroll
        logger.info(f"Bankroll updated: ${old_bankroll:,.2f} -> ${new_bankroll:,.2f} ({change:+.2f})")
    
    def adjust_fractional_factor(self, performance_metrics: Dict[str, float]):
        """
        Dynamically adjust fractional factor based on recent performance
        
        Args:
            performance_metrics: Dictionary with recent performance data
        """
        if "win_rate" in performance_metrics and "sharpe_ratio" in performance_metrics:
            win_rate = performance_metrics["win_rate"]
            sharpe_ratio = performance_metrics.get("sharpe_ratio", 0)
            
            # Increase fractional factor if performing well
            if win_rate > 0.55 and sharpe_ratio > 1.0:
                self.fractional_factor = min(0.5, self.fractional_factor * 1.1)
            # Decrease if performing poorly
            elif win_rate < 0.45 or sharpe_ratio < 0.5:
                self.fractional_factor = max(0.1, self.fractional_factor * 0.9)
            
            logger.info(f"Fractional factor adjusted to: {self.fractional_factor:.3f}")


def calculate_optimal_kelly_fraction(odds: float, 
                                   true_probability: float, 
                                   confidence: float = 1.0) -> float:
    """
    Convenience function to calculate optimal Kelly fraction for a single bet
    
    Args:
        odds: Decimal odds
        true_probability: Estimated true probability of winning
        confidence: Confidence in probability estimate (0-1)
        
    Returns:
        Optimal Kelly fraction
    """
    if odds <= 1.0 or not 0 < true_probability < 1:
        return 0.0
    
    edge = true_probability - (1 / odds)
    if edge <= 0:
        return 0.0
    
    profit_ratio = odds - 1
    kelly_fraction = ((profit_ratio * true_probability) - (1 - true_probability)) / profit_ratio
    
    # Apply confidence adjustment
    kelly_fraction *= confidence
    
    return max(0, kelly_fraction)


# Example usage and testing
if __name__ == "__main__":
    # Create optimizer
    optimizer = KellyCriterionOptimizer(bankroll=10000, fractional_factor=0.25)
    
    # Create sample betting opportunities
    opportunities = [
        BettingOpportunity(
            bet_id="nfl_game_1",
            odds=2.1,
            probability=0.55,
            confidence=0.8,
            sport="NFL",
            event="Chiefs vs Bills"
        ),
        BettingOpportunity(
            bet_id="nba_game_1", 
            odds=1.85,
            probability=0.60,
            confidence=0.9,
            sport="NBA",
            event="Lakers vs Warriors"
        )
    ]
    
    # Calculate optimal allocations
    results = optimizer.optimize_portfolio(opportunities)
    
    # Print results
    for result in results:
        print(f"Bet {result.bet_id}:")
        print(f"  Recommended stake: ${result.recommended_stake:.2f}")
        print(f"  Kelly fraction: {result.kelly_fraction:.3f}")
        print(f"  Expected growth: {result.expected_growth:.4f}")
        print(f"  Warnings: {result.warnings}")
        print()
    
    # Portfolio summary
    summary = optimizer.get_portfolio_summary(results)
    print("Portfolio Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")