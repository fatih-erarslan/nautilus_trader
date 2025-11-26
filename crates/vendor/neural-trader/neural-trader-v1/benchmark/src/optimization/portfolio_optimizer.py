"""Portfolio allocation optimization for multi-strategy trading.

This module provides advanced portfolio optimization techniques including
mean-variance optimization, risk parity, and dynamic allocation strategies.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.stats import norm
import cvxpy as cp
from sklearn.covariance import LedoitWolf, ShrunkCovariance

logger = logging.getLogger(__name__)


class AllocationMethod(Enum):
    """Portfolio allocation methods."""
    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_variance"
    EQUAL_WEIGHT = "equal_weight"
    BLACK_LITTERMAN = "black_litterman"
    HIERARCHICAL_RISK_PARITY = "hrp"
    ROBUST_OPTIMIZATION = "robust"


@dataclass
class PortfolioConstraints:
    """Constraints for portfolio optimization."""
    min_weight: float = 0.0
    max_weight: float = 1.0
    target_volatility: Optional[float] = None
    max_positions: Optional[int] = None
    sector_limits: Optional[Dict[str, float]] = None
    turnover_limit: Optional[float] = None
    leverage_limit: float = 1.0


@dataclass
class OptimizationResult:
    """Portfolio optimization results."""
    weights: np.ndarray
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    diversification_ratio: float
    effective_assets: float
    turnover: Optional[float] = None
    var_95: Optional[float] = None
    cvar_95: Optional[float] = None


@dataclass
class BacktestResult:
    """Portfolio backtest results."""
    returns: pd.Series
    weights_history: pd.DataFrame
    performance_metrics: Dict[str, float]
    turnover_history: pd.Series
    rebalance_dates: List[datetime]


class PortfolioOptimizer:
    """Advanced portfolio optimization engine."""
    
    def __init__(
        self,
        returns_data: pd.DataFrame,
        constraints: Optional[PortfolioConstraints] = None,
        risk_free_rate: float = 0.02
    ):
        """Initialize portfolio optimizer.
        
        Args:
            returns_data: DataFrame of asset returns
            constraints: Portfolio constraints
            risk_free_rate: Risk-free rate for Sharpe calculation
        """
        self.returns_data = returns_data
        self.n_assets = returns_data.shape[1]
        self.asset_names = returns_data.columns.tolist()
        
        self.constraints = constraints or PortfolioConstraints()
        self.risk_free_rate = risk_free_rate
        
        # Precompute statistics
        self._compute_statistics()
        
        logger.info(f"Initialized portfolio optimizer with {self.n_assets} assets")
        
    def _compute_statistics(self):
        """Precompute return statistics."""
        self.expected_returns = self.returns_data.mean().values
        self.covariance_matrix = self.returns_data.cov().values
        
        # Robust covariance estimation
        lw = LedoitWolf()
        self.robust_covariance = lw.fit(self.returns_data).covariance_
        
        # Correlation matrix
        self.correlation_matrix = self.returns_data.corr().values
        
    def optimize(
        self,
        method: Union[AllocationMethod, str] = AllocationMethod.MAX_SHARPE,
        views: Optional[Dict[str, float]] = None,
        confidence: Optional[Dict[str, float]] = None
    ) -> OptimizationResult:
        """Optimize portfolio allocation.
        
        Args:
            method: Optimization method
            views: Market views for Black-Litterman
            confidence: Confidence in views
            
        Returns:
            Optimization results
        """
        if isinstance(method, str):
            method = AllocationMethod(method)
            
        logger.info(f"Optimizing portfolio using {method.value} method")
        
        if method == AllocationMethod.MEAN_VARIANCE:
            weights = self._optimize_mean_variance()
        elif method == AllocationMethod.RISK_PARITY:
            weights = self._optimize_risk_parity()
        elif method == AllocationMethod.MAX_SHARPE:
            weights = self._optimize_max_sharpe()
        elif method == AllocationMethod.MIN_VARIANCE:
            weights = self._optimize_min_variance()
        elif method == AllocationMethod.EQUAL_WEIGHT:
            weights = self._equal_weight()
        elif method == AllocationMethod.BLACK_LITTERMAN:
            weights = self._optimize_black_litterman(views, confidence)
        elif method == AllocationMethod.HIERARCHICAL_RISK_PARITY:
            weights = self._optimize_hrp()
        elif method == AllocationMethod.ROBUST_OPTIMIZATION:
            weights = self._optimize_robust()
        else:
            raise ValueError(f"Unknown optimization method: {method}")
            
        # Calculate portfolio metrics
        result = self._calculate_portfolio_metrics(weights)
        
        return result
        
    def _optimize_mean_variance(self, target_return: Optional[float] = None) -> np.ndarray:
        """Classical mean-variance optimization."""
        # Use cvxpy for convex optimization
        w = cp.Variable(self.n_assets)
        
        # Objective: minimize variance
        variance = cp.quad_form(w, self.covariance_matrix)
        objective = cp.Minimize(variance)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Weights sum to 1
            w >= self.constraints.min_weight,
            w <= self.constraints.max_weight
        ]
        
        # Add target return constraint if specified
        if target_return is not None:
            constraints.append(
                self.expected_returns @ w >= target_return
            )
            
        # Add leverage constraint
        if self.constraints.leverage_limit != 1.0:
            constraints.append(
                cp.norm(w, 1) <= self.constraints.leverage_limit
            )
            
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status != 'optimal':
            logger.warning(f"Optimization status: {problem.status}")
            
        return w.value
        
    def _optimize_risk_parity(self) -> np.ndarray:
        """Risk parity optimization."""
        def risk_parity_objective(weights):
            # Calculate risk contributions
            portfolio_vol = np.sqrt(weights @ self.covariance_matrix @ weights)
            marginal_contrib = self.covariance_matrix @ weights
            contrib = weights * marginal_contrib / portfolio_vol
            
            # Objective: minimize squared differences in risk contributions
            target_contrib = portfolio_vol / self.n_assets
            return np.sum((contrib - target_contrib) ** 2)
            
        # Initial guess: equal weights
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Bounds
        bounds = Bounds(
            lb=np.full(self.n_assets, self.constraints.min_weight),
            ub=np.full(self.n_assets, self.constraints.max_weight)
        )
        
        # Constraint: weights sum to 1
        constraint = LinearConstraint(
            np.ones(self.n_assets),
            lb=1.0,
            ub=1.0
        )
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint
        )
        
        return result.x
        
    def _optimize_max_sharpe(self) -> np.ndarray:
        """Maximum Sharpe ratio optimization."""
        def negative_sharpe(weights):
            portfolio_return = weights @ self.expected_returns
            portfolio_vol = np.sqrt(weights @ self.covariance_matrix @ weights)
            
            # Annualized Sharpe ratio
            sharpe = (portfolio_return - self.risk_free_rate/252) / portfolio_vol
            return -sharpe * np.sqrt(252)
            
        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Bounds
        bounds = Bounds(
            lb=np.full(self.n_assets, self.constraints.min_weight),
            ub=np.full(self.n_assets, self.constraints.max_weight)
        )
        
        # Constraints
        constraints = [
            LinearConstraint(np.ones(self.n_assets), lb=1.0, ub=1.0)
        ]
        
        # Add target volatility constraint if specified
        if self.constraints.target_volatility is not None:
            def volatility_constraint(weights):
                vol = np.sqrt(weights @ self.covariance_matrix @ weights)
                return vol - self.constraints.target_volatility
                
            constraints.append({
                'type': 'eq',
                'fun': volatility_constraint
            })
            
        # Optimize
        result = minimize(
            negative_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
        
    def _optimize_min_variance(self) -> np.ndarray:
        """Minimum variance portfolio."""
        return self._optimize_mean_variance()
        
    def _equal_weight(self) -> np.ndarray:
        """Equal weight portfolio."""
        return np.ones(self.n_assets) / self.n_assets
        
    def _optimize_black_litterman(
        self,
        views: Optional[Dict[str, float]],
        confidence: Optional[Dict[str, float]]
    ) -> np.ndarray:
        """Black-Litterman optimization."""
        # Market equilibrium weights (market cap weighted)
        # For simplicity, use equal weights as market weights
        market_weights = np.ones(self.n_assets) / self.n_assets
        
        # Implied equilibrium returns
        risk_aversion = 2.5
        implied_returns = risk_aversion * self.covariance_matrix @ market_weights
        
        if views is None or not views:
            # No views, return market weights
            return market_weights
            
        # Build views matrix P and views vector Q
        P = []
        Q = []
        omega_diag = []
        
        for asset, view_return in views.items():
            if asset in self.asset_names:
                idx = self.asset_names.index(asset)
                p = np.zeros(self.n_assets)
                p[idx] = 1
                P.append(p)
                Q.append(view_return)
                
                # Uncertainty in view
                conf = confidence.get(asset, 0.5) if confidence else 0.5
                omega_diag.append((1 - conf) * 0.01)  # Scale uncertainty
                
        if not P:
            return market_weights
            
        P = np.array(P)
        Q = np.array(Q)
        Omega = np.diag(omega_diag)
        
        # Black-Litterman formula
        tau = 0.05  # Scaling factor
        
        # Posterior covariance
        M = np.linalg.inv(
            np.linalg.inv(tau * self.covariance_matrix) +
            P.T @ np.linalg.inv(Omega) @ P
        )
        
        # Posterior expected returns
        posterior_returns = M @ (
            np.linalg.inv(tau * self.covariance_matrix) @ implied_returns +
            P.T @ np.linalg.inv(Omega) @ Q
        )
        
        # Use posterior returns in mean-variance optimization
        self.expected_returns = posterior_returns
        weights = self._optimize_max_sharpe()
        
        # Restore original expected returns
        self._compute_statistics()
        
        return weights
        
    def _optimize_hrp(self) -> np.ndarray:
        """Hierarchical Risk Parity optimization."""
        from scipy.cluster.hierarchy import linkage, to_tree
        from scipy.spatial.distance import squareform
        
        # Distance matrix from correlation
        distance_matrix = np.sqrt(0.5 * (1 - self.correlation_matrix))
        condensed_dist = squareform(distance_matrix)
        
        # Hierarchical clustering
        link = linkage(condensed_dist, 'single')
        
        # Get quasi-diagonalization order
        root = to_tree(link)
        order = self._get_quasi_diag(root)
        
        # Recursive bisection
        weights = self._recursive_bisection(
            self.covariance_matrix[np.ix_(order, order)],
            order
        )
        
        # Reorder weights
        reordered_weights = np.zeros(self.n_assets)
        for i, idx in enumerate(order):
            reordered_weights[idx] = weights[i]
            
        return reordered_weights
        
    def _get_quasi_diag(self, node, order=[]):
        """Get quasi-diagonalization order from hierarchical tree."""
        if node.is_leaf():
            order.append(node.id)
        else:
            self._get_quasi_diag(node.left, order)
            self._get_quasi_diag(node.right, order)
        return order
        
    def _recursive_bisection(self, cov, indices):
        """Recursive bisection for HRP."""
        n = cov.shape[0]
        w = np.ones(n)
        
        if n == 1:
            return w
            
        # Split into two clusters
        if n == 2:
            split = 1
        else:
            split = n // 2
            
        # Calculate weights for each cluster
        cov1 = cov[:split, :split]
        cov2 = cov[split:, split:]
        
        # Inverse variance allocation
        ivp1 = 1 / np.diag(cov1).sum()
        ivp2 = 1 / np.diag(cov2).sum()
        
        # Allocate between clusters
        alpha = ivp1 / (ivp1 + ivp2)
        
        # Recursive allocation within clusters
        w[:split] = alpha * self._recursive_bisection(cov1, indices[:split])
        w[split:] = (1 - alpha) * self._recursive_bisection(cov2, indices[split:])
        
        return w
        
    def _optimize_robust(self) -> np.ndarray:
        """Robust optimization with uncertainty."""
        # Use robust covariance matrix
        w = cp.Variable(self.n_assets)
        
        # Worst-case variance with uncertainty set
        uncertainty_level = 0.1  # 10% uncertainty
        
        # Robust objective: minimize worst-case variance
        variance = cp.quad_form(w, self.robust_covariance)
        
        # Add uncertainty penalty
        uncertainty_penalty = uncertainty_level * cp.norm(w, 2)
        
        objective = cp.Minimize(variance + uncertainty_penalty)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= self.constraints.min_weight,
            w <= self.constraints.max_weight
        ]
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        return w.value
        
    def _calculate_portfolio_metrics(self, weights: np.ndarray) -> OptimizationResult:
        """Calculate portfolio performance metrics."""
        # Expected return and volatility
        portfolio_return = weights @ self.expected_returns
        portfolio_variance = weights @ self.covariance_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # Annualized metrics
        annual_return = portfolio_return * 252
        annual_vol = portfolio_vol * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_vol
        
        # Diversification ratio
        weighted_vol = weights @ np.sqrt(np.diag(self.covariance_matrix))
        diversification_ratio = weighted_vol / portfolio_vol
        
        # Effective number of assets (inverse HHI)
        hhi = np.sum(weights ** 2)
        effective_assets = 1 / hhi if hhi > 0 else self.n_assets
        
        # Value at Risk (95% confidence)
        var_95 = norm.ppf(0.05) * portfolio_vol * np.sqrt(252)
        
        # Conditional Value at Risk
        cvar_95 = -portfolio_return + portfolio_vol * norm.pdf(norm.ppf(0.05)) / 0.05
        cvar_95 *= np.sqrt(252)
        
        return OptimizationResult(
            weights=weights,
            expected_return=annual_return,
            expected_volatility=annual_vol,
            sharpe_ratio=sharpe_ratio,
            diversification_ratio=diversification_ratio,
            effective_assets=effective_assets,
            var_95=var_95,
            cvar_95=cvar_95
        )
        
    def backtest(
        self,
        method: AllocationMethod,
        rebalance_frequency: str = 'monthly',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        transaction_cost: float = 0.001
    ) -> BacktestResult:
        """Backtest portfolio allocation strategy.
        
        Args:
            method: Allocation method
            rebalance_frequency: 'daily', 'weekly', 'monthly', 'quarterly'
            start_date: Backtest start date
            end_date: Backtest end date
            transaction_cost: Transaction cost per trade
            
        Returns:
            Backtest results
        """
        # Set date range
        if start_date is None:
            start_date = self.returns_data.index[0]
        if end_date is None:
            end_date = self.returns_data.index[-1]
            
        # Filter data
        backtest_data = self.returns_data.loc[start_date:end_date]
        
        # Determine rebalance dates
        rebalance_dates = self._get_rebalance_dates(
            backtest_data.index,
            rebalance_frequency
        )
        
        # Initialize tracking
        portfolio_returns = []
        weights_history = []
        turnover_history = []
        current_weights = None
        
        # Run backtest
        for i, date in enumerate(backtest_data.index):
            if date in rebalance_dates or current_weights is None:
                # Rebalance portfolio
                lookback_data = backtest_data.loc[:date].iloc[-252:]  # 1 year lookback
                
                # Create temporary optimizer with lookback data
                temp_optimizer = PortfolioOptimizer(
                    lookback_data,
                    self.constraints,
                    self.risk_free_rate
                )
                
                # Optimize weights
                result = temp_optimizer.optimize(method)
                new_weights = result.weights
                
                # Calculate turnover
                if current_weights is not None:
                    turnover = np.sum(np.abs(new_weights - current_weights))
                    turnover_cost = turnover * transaction_cost
                else:
                    turnover = 0
                    turnover_cost = 0
                    
                current_weights = new_weights
                turnover_history.append(turnover)
                
            # Calculate portfolio return
            daily_returns = backtest_data.iloc[i].values
            portfolio_return = np.sum(current_weights * daily_returns) - turnover_cost
            portfolio_returns.append(portfolio_return)
            
            # Track weights
            weights_dict = {
                asset: weight 
                for asset, weight in zip(self.asset_names, current_weights)
            }
            weights_dict['date'] = date
            weights_history.append(weights_dict)
            
        # Create results DataFrames
        returns_series = pd.Series(
            portfolio_returns,
            index=backtest_data.index,
            name='portfolio_returns'
        )
        
        weights_df = pd.DataFrame(weights_history).set_index('date')
        
        turnover_series = pd.Series(
            turnover_history,
            index=rebalance_dates,
            name='turnover'
        )
        
        # Calculate performance metrics
        performance_metrics = self._calculate_backtest_metrics(returns_series)
        
        return BacktestResult(
            returns=returns_series,
            weights_history=weights_df,
            performance_metrics=performance_metrics,
            turnover_history=turnover_series,
            rebalance_dates=rebalance_dates
        )
        
    def _get_rebalance_dates(
        self,
        dates: pd.DatetimeIndex,
        frequency: str
    ) -> List[datetime]:
        """Get rebalancing dates based on frequency."""
        if frequency == 'daily':
            return dates.tolist()
        elif frequency == 'weekly':
            # Every Monday
            return dates[dates.dayofweek == 0].tolist()
        elif frequency == 'monthly':
            # First trading day of month
            return dates.to_series().groupby(
                pd.Grouper(freq='M')
            ).first().dropna().index.tolist()
        elif frequency == 'quarterly':
            # First trading day of quarter
            return dates.to_series().groupby(
                pd.Grouper(freq='Q')
            ).first().dropna().index.tolist()
        else:
            raise ValueError(f"Unknown rebalance frequency: {frequency}")
            
    def _calculate_backtest_metrics(
        self,
        returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate backtest performance metrics."""
        # Cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Total return
        total_return = cumulative_returns.iloc[-1] - 1
        
        # Annualized return
        n_years = len(returns) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1
        
        # Volatility
        annual_vol = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_vol
        
        # Maximum drawdown
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_vol
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': (returns > 0).mean(),
            'best_day': returns.max(),
            'worst_day': returns.min()
        }


class DynamicPortfolioManager:
    """Dynamic portfolio management with regime detection."""
    
    def __init__(
        self,
        optimizer: PortfolioOptimizer,
        regime_detector: Optional[Any] = None
    ):
        """Initialize dynamic portfolio manager.
        
        Args:
            optimizer: Base portfolio optimizer
            regime_detector: Market regime detection model
        """
        self.optimizer = optimizer
        self.regime_detector = regime_detector
        
        # Regime-specific strategies
        self.regime_strategies = {
            'bull_market': AllocationMethod.MAX_SHARPE,
            'bear_market': AllocationMethod.MIN_VARIANCE,
            'high_volatility': AllocationMethod.RISK_PARITY,
            'low_volatility': AllocationMethod.MAX_SHARPE,
            'crisis': AllocationMethod.MIN_VARIANCE
        }
        
    def allocate(
        self,
        current_data: pd.DataFrame,
        current_regime: Optional[str] = None
    ) -> OptimizationResult:
        """Dynamically allocate portfolio based on market conditions.
        
        Args:
            current_data: Current market data
            current_regime: Detected market regime
            
        Returns:
            Dynamic allocation result
        """
        # Detect regime if not provided
        if current_regime is None and self.regime_detector is not None:
            current_regime = self.regime_detector.detect(current_data)
            
        # Select allocation method based on regime
        if current_regime in self.regime_strategies:
            method = self.regime_strategies[current_regime]
        else:
            method = AllocationMethod.RISK_PARITY  # Default
            
        logger.info(f"Allocating for {current_regime} regime using {method.value}")
        
        # Adjust constraints for regime
        adjusted_constraints = self._adjust_constraints_for_regime(
            current_regime
        )
        
        # Update optimizer constraints
        original_constraints = self.optimizer.constraints
        self.optimizer.constraints = adjusted_constraints
        
        # Optimize
        result = self.optimizer.optimize(method)
        
        # Restore original constraints
        self.optimizer.constraints = original_constraints
        
        return result
        
    def _adjust_constraints_for_regime(
        self,
        regime: str
    ) -> PortfolioConstraints:
        """Adjust portfolio constraints based on regime."""
        base_constraints = self.optimizer.constraints
        
        if regime == 'crisis':
            # More conservative in crisis
            return PortfolioConstraints(
                min_weight=0.0,
                max_weight=0.3,  # Lower concentration
                target_volatility=base_constraints.target_volatility * 0.5,
                leverage_limit=0.8  # Reduce leverage
            )
        elif regime == 'high_volatility':
            # Spread risk in high volatility
            return PortfolioConstraints(
                min_weight=0.02,  # Minimum position
                max_weight=0.4,
                target_volatility=base_constraints.target_volatility * 0.7
            )
        elif regime == 'bull_market':
            # Allow more concentration in bull market
            return PortfolioConstraints(
                min_weight=0.0,
                max_weight=0.6,
                leverage_limit=1.2  # Allow some leverage
            )
        else:
            return base_constraints