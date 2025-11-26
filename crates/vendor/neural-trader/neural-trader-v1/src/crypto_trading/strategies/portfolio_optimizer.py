"""
Portfolio Optimizer

Advanced portfolio optimization algorithms including Markowitz optimization,
Kelly criterion, and risk parity approaches.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy.optimize import minimize, LinearConstraint
from scipy.stats import norm
import pandas as pd
from datetime import datetime, timedelta


@dataclass
class AssetData:
    """Historical data for an asset"""
    symbol: str
    returns: np.ndarray
    prices: np.ndarray
    volumes: np.ndarray
    timestamps: List[datetime]
    
    @property
    def mean_return(self) -> float:
        """Average return"""
        return np.mean(self.returns)
    
    @property
    def volatility(self) -> float:
        """Standard deviation of returns"""
        return np.std(self.returns)
    
    @property
    def sharpe_ratio(self) -> float:
        """Sharpe ratio (assuming 0 risk-free rate)"""
        if self.volatility == 0:
            return 0
        return self.mean_return / self.volatility * np.sqrt(365)


@dataclass
class OptimizationConstraints:
    """Constraints for portfolio optimization"""
    min_weight: float = 0.0
    max_weight: float = 1.0
    min_positions: int = 1
    max_positions: int = 50
    target_return: Optional[float] = None
    max_volatility: Optional[float] = None
    sector_limits: Dict[str, float] = None
    
    def __post_init__(self):
        if self.sector_limits is None:
            self.sector_limits = {}


@dataclass
class OptimizationResult:
    """Result of portfolio optimization"""
    weights: np.ndarray
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    value_at_risk: float
    conditional_value_at_risk: float
    diversification_ratio: float
    effective_assets: float
    turnover: Optional[float] = None


class PortfolioOptimizer:
    """Advanced portfolio optimization methods"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize portfolio optimizer
        
        Args:
            risk_free_rate: Annual risk-free rate for calculations
        """
        self.risk_free_rate = risk_free_rate
        self.daily_rf = risk_free_rate / 365
        
    def markowitz_optimization(self,
                             assets: List[AssetData],
                             constraints: OptimizationConstraints,
                             target: str = 'sharpe') -> OptimizationResult:
        """
        Markowitz mean-variance optimization
        
        Args:
            assets: List of asset data
            constraints: Optimization constraints
            target: Optimization target ('sharpe', 'return', 'volatility')
            
        Returns:
            Optimization result
        """
        n_assets = len(assets)
        
        # Calculate returns matrix and covariance
        returns_matrix = np.array([asset.returns for asset in assets]).T
        mean_returns = np.array([asset.mean_return for asset in assets])
        cov_matrix = np.cov(returns_matrix, rowvar=False)
        
        # Define objective functions
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            if portfolio_vol == 0:
                return 0
            return -(portfolio_return - self.daily_rf) / portfolio_vol * np.sqrt(365)
        
        def negative_return(weights):
            return -np.dot(weights, mean_returns)
        
        def volatility(weights):
            return np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        
        # Select objective
        objectives = {
            'sharpe': negative_sharpe,
            'return': negative_return,
            'volatility': volatility
        }
        objective = objectives.get(target, negative_sharpe)
        
        # Set up constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Add optional constraints
        if constraints.target_return is not None:
            constraints_list.append({
                'type': 'ineq',
                'fun': lambda w: np.dot(w, mean_returns) - constraints.target_return
            })
            
        if constraints.max_volatility is not None:
            constraints_list.append({
                'type': 'ineq',
                'fun': lambda w: constraints.max_volatility - volatility(w)
            })
        
        # Bounds for weights
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        initial_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        if not result.success:
            # Fallback to equal weights
            optimal_weights = initial_weights
        else:
            optimal_weights = result.x
            
        # Calculate portfolio metrics
        return self._calculate_portfolio_metrics(optimal_weights, assets, returns_matrix, cov_matrix)
    
    def risk_parity_optimization(self,
                               assets: List[AssetData],
                               constraints: OptimizationConstraints) -> OptimizationResult:
        """
        Risk parity optimization - equal risk contribution from each asset
        
        Args:
            assets: List of asset data
            constraints: Optimization constraints
            
        Returns:
            Optimization result
        """
        n_assets = len(assets)
        
        # Calculate returns and covariance
        returns_matrix = np.array([asset.returns for asset in assets]).T
        mean_returns = np.array([asset.mean_return for asset in assets])
        cov_matrix = np.cov(returns_matrix, rowvar=False)
        
        # Risk parity objective
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            # Marginal risk contributions
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            
            # Risk contributions
            contrib = weights * marginal_contrib
            
            # Target equal contribution
            target_contrib = portfolio_vol / n_assets
            
            # Sum of squared differences from target
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Bounds
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Initial guess (inverse volatility weighting)
        volatilities = np.array([asset.volatility for asset in assets])
        inv_vol = 1 / volatilities
        initial_weights = inv_vol / np.sum(inv_vol)
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        optimal_weights = result.x if result.success else initial_weights
        
        return self._calculate_portfolio_metrics(optimal_weights, assets, returns_matrix, cov_matrix)
    
    def kelly_criterion(self,
                       assets: List[AssetData],
                       constraints: OptimizationConstraints,
                       confidence: float = 0.25) -> OptimizationResult:
        """
        Kelly criterion optimization for optimal growth
        
        Args:
            assets: List of asset data
            constraints: Optimization constraints
            confidence: Confidence factor (fraction of Kelly to use)
            
        Returns:
            Optimization result
        """
        n_assets = len(assets)
        
        # Calculate returns and covariance
        returns_matrix = np.array([asset.returns for asset in assets]).T
        mean_returns = np.array([asset.mean_return for asset in assets])
        cov_matrix = np.cov(returns_matrix, rowvar=False)
        
        # Add small regularization to avoid singularity
        cov_matrix += np.eye(n_assets) * 1e-8
        
        try:
            # Kelly formula: f = (μ - r) / σ²
            # For multiple assets: w = Σ^(-1) * (μ - r)
            excess_returns = mean_returns - self.daily_rf
            
            # Inverse covariance
            inv_cov = np.linalg.inv(cov_matrix)
            
            # Raw Kelly weights
            kelly_weights = np.dot(inv_cov, excess_returns)
            
            # Apply confidence factor
            kelly_weights *= confidence
            
            # Normalize to sum to 1 (if positive)
            positive_weights = np.maximum(kelly_weights, 0)
            if np.sum(positive_weights) > 0:
                kelly_weights = positive_weights / np.sum(positive_weights)
            else:
                # Fallback to equal weights
                kelly_weights = np.ones(n_assets) / n_assets
                
        except np.linalg.LinAlgError:
            # Fallback for singular matrix
            kelly_weights = np.ones(n_assets) / n_assets
        
        # Apply constraints
        kelly_weights = np.clip(kelly_weights, constraints.min_weight, constraints.max_weight)
        kelly_weights = kelly_weights / np.sum(kelly_weights)
        
        return self._calculate_portfolio_metrics(kelly_weights, assets, returns_matrix, cov_matrix)
    
    def maximum_diversification(self,
                              assets: List[AssetData],
                              constraints: OptimizationConstraints) -> OptimizationResult:
        """
        Maximum diversification optimization
        
        Args:
            assets: List of asset data
            constraints: Optimization constraints
            
        Returns:
            Optimization result
        """
        n_assets = len(assets)
        
        # Calculate returns and covariance
        returns_matrix = np.array([asset.returns for asset in assets]).T
        mean_returns = np.array([asset.mean_return for asset in assets])
        cov_matrix = np.cov(returns_matrix, rowvar=False)
        volatilities = np.array([asset.volatility for asset in assets])
        
        # Diversification ratio objective (to maximize)
        def negative_div_ratio(weights):
            weighted_avg_vol = np.dot(weights, volatilities)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            if portfolio_vol == 0:
                return 0
            return -weighted_avg_vol / portfolio_vol
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Bounds
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Initial guess
        initial_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            negative_div_ratio,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        optimal_weights = result.x if result.success else initial_weights
        
        return self._calculate_portfolio_metrics(optimal_weights, assets, returns_matrix, cov_matrix)
    
    def hierarchical_risk_parity(self,
                               assets: List[AssetData],
                               constraints: OptimizationConstraints) -> OptimizationResult:
        """
        Hierarchical Risk Parity (HRP) optimization
        
        Args:
            assets: List of asset data
            constraints: Optimization constraints
            
        Returns:
            Optimization result
        """
        n_assets = len(assets)
        
        # Calculate returns and correlation
        returns_matrix = np.array([asset.returns for asset in assets]).T
        mean_returns = np.array([asset.mean_return for asset in assets])
        cov_matrix = np.cov(returns_matrix, rowvar=False)
        corr_matrix = np.corrcoef(returns_matrix, rowvar=False)
        
        # Distance matrix
        distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))
        
        # Perform hierarchical clustering
        clusters = self._hierarchical_clustering(distance_matrix)
        
        # Allocate weights using inverse variance
        weights = self._hrp_allocation(cov_matrix, clusters)
        
        # Apply constraints
        weights = np.clip(weights, constraints.min_weight, constraints.max_weight)
        weights = weights / np.sum(weights)
        
        return self._calculate_portfolio_metrics(weights, assets, returns_matrix, cov_matrix)
    
    def _calculate_portfolio_metrics(self,
                                   weights: np.ndarray,
                                   assets: List[AssetData],
                                   returns_matrix: np.ndarray,
                                   cov_matrix: np.ndarray) -> OptimizationResult:
        """Calculate comprehensive portfolio metrics"""
        # Basic metrics
        mean_returns = np.array([asset.mean_return for asset in assets])
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        
        # Sharpe ratio
        sharpe = (portfolio_return - self.daily_rf) / portfolio_vol * np.sqrt(365) if portfolio_vol > 0 else 0
        
        # Portfolio returns
        portfolio_returns = np.dot(returns_matrix, weights)
        
        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < self.daily_rf]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else portfolio_vol
        sortino = (portfolio_return - self.daily_rf) / downside_deviation * np.sqrt(365) if downside_deviation > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns))
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(portfolio_returns, 5)
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])
        
        # Diversification ratio
        volatilities = np.array([asset.volatility for asset in assets])
        weighted_avg_vol = np.dot(weights, volatilities)
        div_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1
        
        # Effective number of assets (inverse HHI)
        hhi = np.sum(weights ** 2)
        effective_assets = 1 / hhi if hhi > 0 else 1
        
        return OptimizationResult(
            weights=weights,
            expected_return=portfolio_return * 365,  # Annualized
            expected_volatility=portfolio_vol * np.sqrt(365),  # Annualized
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            value_at_risk=var_95,
            conditional_value_at_risk=cvar_95,
            diversification_ratio=div_ratio,
            effective_assets=effective_assets
        )
    
    def _hierarchical_clustering(self, distance_matrix: np.ndarray) -> List[List[int]]:
        """Simple hierarchical clustering implementation"""
        n = len(distance_matrix)
        clusters = [[i] for i in range(n)]
        
        # Simplified clustering - would use scipy.cluster.hierarchy in production
        while len(clusters) > 1:
            # Find closest clusters
            min_dist = float('inf')
            merge_i, merge_j = 0, 1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Average linkage distance
                    dist = np.mean([distance_matrix[a, b] 
                                   for a in clusters[i] 
                                   for b in clusters[j]])
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j
            
            # Merge clusters
            clusters[merge_i].extend(clusters[merge_j])
            clusters.pop(merge_j)
            
        return clusters
    
    def _hrp_allocation(self, cov_matrix: np.ndarray, clusters: List[List[int]]) -> np.ndarray:
        """HRP allocation within clusters"""
        n = len(cov_matrix)
        weights = np.zeros(n)
        
        # Simple inverse variance weighting
        for i in range(n):
            weights[i] = 1 / cov_matrix[i, i]
            
        # Normalize
        weights = weights / np.sum(weights)
        
        return weights
    
    def efficient_frontier(self,
                         assets: List[AssetData],
                         constraints: OptimizationConstraints,
                         n_portfolios: int = 50) -> List[OptimizationResult]:
        """
        Calculate efficient frontier
        
        Args:
            assets: List of asset data
            constraints: Optimization constraints
            n_portfolios: Number of portfolios on frontier
            
        Returns:
            List of optimization results forming the frontier
        """
        # Get minimum variance portfolio
        min_var_result = self.markowitz_optimization(assets, constraints, target='volatility')
        
        # Get maximum return portfolio
        max_ret_constraints = constraints
        max_ret_result = self.markowitz_optimization(assets, constraints, target='return')
        
        # Generate target returns
        min_return = min_var_result.expected_return / 365  # Daily
        max_return = max_ret_result.expected_return / 365  # Daily
        target_returns = np.linspace(min_return, max_return, n_portfolios)
        
        # Calculate frontier
        frontier = []
        for target_ret in target_returns:
            const_copy = OptimizationConstraints(
                min_weight=constraints.min_weight,
                max_weight=constraints.max_weight,
                target_return=target_ret
            )
            result = self.markowitz_optimization(assets, const_copy, target='volatility')
            frontier.append(result)
            
        return frontier
    
    def rebalancing_analysis(self,
                           current_weights: np.ndarray,
                           target_weights: np.ndarray,
                           transaction_costs: float = 0.001) -> Dict[str, float]:
        """
        Analyze rebalancing from current to target weights
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            transaction_costs: Transaction cost rate
            
        Returns:
            Dictionary with rebalancing metrics
        """
        # Calculate turnover
        turnover = np.sum(np.abs(target_weights - current_weights)) / 2
        
        # Estimated transaction costs
        total_costs = turnover * transaction_costs
        
        # Number of trades needed
        trades_needed = np.sum(target_weights != current_weights)
        
        # Maximum position change
        max_change = np.max(np.abs(target_weights - current_weights))
        
        return {
            'turnover': turnover,
            'transaction_costs': total_costs,
            'trades_needed': trades_needed,
            'max_position_change': max_change,
            'total_weight_change': np.sum(np.abs(target_weights - current_weights))
        }