"""
Portfolio Risk Management for Sports Betting

Implements Kelly criterion, fractional Kelly adjustments, multi-sport portfolio optimization,
and correlation analysis between bets.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings


class BettingStrategy(Enum):
    """Betting strategy types"""
    FULL_KELLY = "full_kelly"
    FRACTIONAL_KELLY = "fractional_kelly"
    FIXED_FRACTION = "fixed_fraction"
    PROPORTIONAL = "proportional"
    MINIMUM_VARIANCE = "minimum_variance"


@dataclass
class BetOpportunity:
    """Represents a betting opportunity"""
    bet_id: str
    sport: str
    event: str
    selection: str
    odds: float
    probability: float  # Estimated true probability
    edge: Optional[float] = None
    confidence: float = 1.0
    correlation_group: Optional[str] = None
    
    def __post_init__(self):
        """Calculate edge if not provided"""
        if self.edge is None:
            decimal_odds = self.odds
            implied_prob = 1 / decimal_odds
            self.edge = self.probability - implied_prob


@dataclass
class PortfolioAllocation:
    """Portfolio allocation result"""
    bet_id: str
    allocation_percentage: float
    kelly_percentage: float
    adjusted_percentage: float
    risk_contribution: float
    expected_return: float
    confidence_adjusted: bool


class PortfolioRiskManager:
    """
    Manages portfolio risk for sports betting using Kelly criterion
    and advanced portfolio optimization techniques.
    """
    
    def __init__(self,
                 bankroll: float,
                 max_kelly_fraction: float = 0.25,
                 max_portfolio_risk: float = 0.10,
                 correlation_threshold: float = 0.5,
                 min_edge_threshold: float = 0.02):
        """
        Initialize Portfolio Risk Manager
        
        Args:
            bankroll: Total available bankroll
            max_kelly_fraction: Maximum fraction of Kelly to use (default 0.25 = 25% of Kelly)
            max_portfolio_risk: Maximum portfolio risk as fraction of bankroll
            correlation_threshold: Threshold for considering bets correlated
            min_edge_threshold: Minimum edge required to consider a bet
        """
        self.bankroll = bankroll
        self.max_kelly_fraction = max_kelly_fraction
        self.max_portfolio_risk = max_portfolio_risk
        self.correlation_threshold = correlation_threshold
        self.min_edge_threshold = min_edge_threshold
        self.bet_history = []
        self.correlation_matrix = {}
        
    def calculate_kelly_percentage(self, bet: BetOpportunity) -> float:
        """
        Calculate Kelly percentage for a single bet
        
        Args:
            bet: Betting opportunity
            
        Returns:
            Kelly percentage (0-1)
        """
        if bet.edge <= 0:
            return 0.0
            
        # Kelly formula: f* = (p*b - q) / b
        # where p = probability of win, q = probability of loss, b = odds - 1
        p = bet.probability
        q = 1 - p
        b = bet.odds - 1
        
        kelly = (p * b - q) / b
        
        # Apply confidence adjustment
        kelly *= bet.confidence
        
        # Ensure non-negative and apply sanity limits
        kelly = max(0, min(kelly, 1.0))
        
        return kelly
    
    def calculate_fractional_kelly(self,
                                   bets: List[BetOpportunity],
                                   strategy: BettingStrategy = BettingStrategy.FRACTIONAL_KELLY
                                   ) -> Dict[str, float]:
        """
        Calculate fractional Kelly allocations for multiple bets
        
        Args:
            bets: List of betting opportunities
            strategy: Betting strategy to use
            
        Returns:
            Dictionary of bet_id to allocation percentage
        """
        allocations = {}
        
        # Filter bets by minimum edge
        valid_bets = [bet for bet in bets if bet.edge >= self.min_edge_threshold]
        
        if not valid_bets:
            return allocations
            
        # Calculate Kelly for each bet
        for bet in valid_bets:
            kelly = self.calculate_kelly_percentage(bet)
            
            if strategy == BettingStrategy.FULL_KELLY:
                allocation = kelly
            elif strategy == BettingStrategy.FRACTIONAL_KELLY:
                allocation = kelly * self.max_kelly_fraction
            elif strategy == BettingStrategy.FIXED_FRACTION:
                # Fixed fraction regardless of edge
                allocation = 0.02  # 2% per bet
            else:
                allocation = kelly * self.max_kelly_fraction
                
            allocations[bet.bet_id] = allocation
            
        return allocations
    
    def optimize_multi_sport_portfolio(self,
                                       bets: List[BetOpportunity],
                                       correlation_matrix: Optional[np.ndarray] = None
                                       ) -> List[PortfolioAllocation]:
        """
        Optimize portfolio allocation across multiple sports/bets considering correlations
        
        Args:
            bets: List of betting opportunities
            correlation_matrix: Optional correlation matrix between bets
            
        Returns:
            List of portfolio allocations
        """
        if not bets:
            return []
            
        n_bets = len(bets)
        
        # Calculate base Kelly allocations
        kelly_allocations = self.calculate_fractional_kelly(bets)
        
        # Build correlation matrix if not provided
        if correlation_matrix is None:
            correlation_matrix = self._estimate_correlation_matrix(bets)
            
        # Calculate expected returns and covariance
        expected_returns = np.array([bet.edge for bet in bets])
        kelly_weights = np.array([kelly_allocations.get(bet.bet_id, 0) for bet in bets])
        
        # Apply correlation adjustments
        adjusted_weights = self._apply_correlation_adjustment(
            kelly_weights, correlation_matrix, expected_returns
        )
        
        # Apply portfolio risk constraint
        adjusted_weights = self._apply_risk_constraint(adjusted_weights, correlation_matrix)
        
        # Normalize to ensure sum doesn't exceed max portfolio risk
        total_allocation = np.sum(adjusted_weights)
        if total_allocation > self.max_portfolio_risk:
            adjusted_weights *= self.max_portfolio_risk / total_allocation
            
        # Create allocation results
        allocations = []
        for i, bet in enumerate(bets):
            if adjusted_weights[i] > 0:
                allocation = PortfolioAllocation(
                    bet_id=bet.bet_id,
                    allocation_percentage=adjusted_weights[i],
                    kelly_percentage=kelly_allocations.get(bet.bet_id, 0),
                    adjusted_percentage=adjusted_weights[i],
                    risk_contribution=self._calculate_risk_contribution(
                        i, adjusted_weights, correlation_matrix
                    ),
                    expected_return=bet.edge * adjusted_weights[i],
                    confidence_adjusted=bet.confidence < 1.0
                )
                allocations.append(allocation)
                
        return allocations
    
    def _estimate_correlation_matrix(self, bets: List[BetOpportunity]) -> np.ndarray:
        """
        Estimate correlation matrix between bets based on sport, league, and timing
        
        Args:
            bets: List of betting opportunities
            
        Returns:
            Correlation matrix
        """
        n = len(bets)
        correlation_matrix = np.eye(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                correlation = 0.0
                
                # Same sport correlation
                if bets[i].sport == bets[j].sport:
                    correlation += 0.3
                    
                # Same event correlation
                if bets[i].event == bets[j].event:
                    correlation += 0.5
                    
                # Same correlation group (e.g., same league, tournament)
                if (bets[i].correlation_group and 
                    bets[i].correlation_group == bets[j].correlation_group):
                    correlation += 0.2
                    
                # Cap correlation
                correlation = min(correlation, 0.9)
                
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
                
        return correlation_matrix
    
    def _apply_correlation_adjustment(self,
                                      kelly_weights: np.ndarray,
                                      correlation_matrix: np.ndarray,
                                      expected_returns: np.ndarray
                                      ) -> np.ndarray:
        """
        Adjust weights based on correlations to reduce concentration risk
        
        Args:
            kelly_weights: Initial Kelly-based weights
            correlation_matrix: Correlation matrix
            expected_returns: Expected returns for each bet
            
        Returns:
            Adjusted weights
        """
        # Calculate portfolio variance contribution of each bet
        portfolio_variance = kelly_weights @ correlation_matrix @ kelly_weights
        
        if portfolio_variance > 0:
            # Calculate marginal contribution to risk
            marginal_risk = 2 * correlation_matrix @ kelly_weights / portfolio_variance
            
            # Adjust weights based on risk-return ratio
            risk_adjusted_returns = expected_returns / (marginal_risk + 0.001)
            
            # Scale weights
            scaling_factors = risk_adjusted_returns / np.max(risk_adjusted_returns)
            adjusted_weights = kelly_weights * scaling_factors
        else:
            adjusted_weights = kelly_weights
            
        return adjusted_weights
    
    def _apply_risk_constraint(self,
                               weights: np.ndarray,
                               correlation_matrix: np.ndarray
                               ) -> np.ndarray:
        """
        Apply portfolio risk constraint to ensure total risk is within limits
        
        Args:
            weights: Current weights
            correlation_matrix: Correlation matrix
            
        Returns:
            Risk-constrained weights
        """
        # Calculate portfolio standard deviation
        portfolio_variance = weights @ correlation_matrix @ weights
        portfolio_std = np.sqrt(portfolio_variance)
        
        # If risk exceeds limit, scale down proportionally
        if portfolio_std > self.max_portfolio_risk:
            scaling_factor = self.max_portfolio_risk / portfolio_std
            weights *= scaling_factor
            
        return weights
    
    def _calculate_risk_contribution(self,
                                     bet_index: int,
                                     weights: np.ndarray,
                                     correlation_matrix: np.ndarray
                                     ) -> float:
        """
        Calculate risk contribution of a specific bet to the portfolio
        
        Args:
            bet_index: Index of the bet
            weights: Portfolio weights
            correlation_matrix: Correlation matrix
            
        Returns:
            Risk contribution (0-1)
        """
        portfolio_variance = weights @ correlation_matrix @ weights
        
        if portfolio_variance > 0:
            # Marginal contribution to risk
            marginal_contribution = correlation_matrix[bet_index] @ weights
            risk_contribution = weights[bet_index] * marginal_contribution / portfolio_variance
        else:
            risk_contribution = 0.0
            
        return risk_contribution
    
    def calculate_portfolio_metrics(self,
                                    allocations: List[PortfolioAllocation],
                                    bets: List[BetOpportunity]
                                    ) -> Dict[str, float]:
        """
        Calculate portfolio-level risk metrics
        
        Args:
            allocations: List of portfolio allocations
            bets: List of betting opportunities
            
        Returns:
            Dictionary of portfolio metrics
        """
        if not allocations:
            return {
                'total_allocation': 0.0,
                'expected_return': 0.0,
                'portfolio_variance': 0.0,
                'sharpe_ratio': 0.0,
                'max_bet_allocation': 0.0,
                'diversification_ratio': 0.0
            }
            
        # Create weight vector
        bet_dict = {bet.bet_id: bet for bet in bets}
        weights = np.zeros(len(bets))
        returns = np.zeros(len(bets))
        
        for alloc in allocations:
            if alloc.bet_id in bet_dict:
                idx = next(i for i, bet in enumerate(bets) if bet.bet_id == alloc.bet_id)
                weights[idx] = alloc.allocation_percentage
                returns[idx] = bet_dict[alloc.bet_id].edge
                
        # Calculate metrics
        total_allocation = np.sum(weights)
        expected_return = np.sum(weights * returns)
        
        # Portfolio variance
        correlation_matrix = self._estimate_correlation_matrix(bets)
        portfolio_variance = weights @ correlation_matrix @ weights
        portfolio_std = np.sqrt(portfolio_variance)
        
        # Sharpe ratio (assuming risk-free rate = 0 for betting)
        sharpe_ratio = expected_return / portfolio_std if portfolio_std > 0 else 0
        
        # Diversification metrics
        max_bet_allocation = np.max(weights) if len(weights) > 0 else 0
        non_zero_bets = np.sum(weights > 0)
        diversification_ratio = non_zero_bets / len(bets) if len(bets) > 0 else 0
        
        return {
            'total_allocation': total_allocation,
            'expected_return': expected_return,
            'portfolio_variance': portfolio_variance,
            'portfolio_std': portfolio_std,
            'sharpe_ratio': sharpe_ratio,
            'max_bet_allocation': max_bet_allocation,
            'diversification_ratio': diversification_ratio,
            'number_of_bets': non_zero_bets
        }
    
    def update_bankroll(self, amount: float):
        """Update bankroll amount"""
        self.bankroll = amount
        
    def get_bet_size(self, allocation_percentage: float) -> float:
        """Calculate actual bet size from allocation percentage"""
        return self.bankroll * allocation_percentage