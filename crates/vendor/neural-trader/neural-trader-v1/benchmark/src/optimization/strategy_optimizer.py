"""Strategy-specific optimization for trading strategies.

This module provides specialized optimization routines for different types
of trading strategies with walk-forward analysis and ensemble methods.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import TimeSeriesSplit
from copy import deepcopy

from .optimizer import OptimizationEngine, OptimizationResult
from .parameter_search import ParameterSpace, HyperparameterSearch
from ..config import OptimizationConfig


logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types of trading strategies."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    TREND_FOLLOWING = "trend_following"
    SENTIMENT_BASED = "sentiment_based"
    ML_BASED = "ml_based"
    HYBRID = "hybrid"


@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis."""
    in_sample_results: List[Dict[str, Any]]
    out_of_sample_results: List[Dict[str, Any]]
    parameter_stability: Dict[str, float]
    performance_consistency: float
    robust_parameters: Dict[str, Any]
    sharpe_degradation: float


@dataclass
class EnsembleResult:
    """Results from ensemble strategy optimization."""
    member_strategies: List[Dict[str, Any]]
    ensemble_weights: List[float]
    individual_performance: List[float]
    ensemble_performance: float
    correlation_matrix: np.ndarray
    diversity_score: float


class StrategyOptimizer:
    """Optimizes trading strategies with advanced techniques."""
    
    def __init__(
        self,
        strategy_class: Any,
        data: pd.DataFrame,
        config: OptimizationConfig
    ):
        """Initialize strategy optimizer.
        
        Args:
            strategy_class: Trading strategy class to optimize
            data: Historical market data
            config: Optimization configuration
        """
        self.strategy_class = strategy_class
        self.data = data
        self.config = config
        
        # Detect strategy type
        self.strategy_type = self._detect_strategy_type()
        
        # Strategy evaluator
        self.evaluator = StrategyEvaluator(data)
        
        logger.info(f"Initialized optimizer for {self.strategy_type.value} strategy")
        
    def _detect_strategy_type(self) -> StrategyType:
        """Detect strategy type from class attributes."""
        # Check for strategy type hints in class
        if hasattr(self.strategy_class, 'STRATEGY_TYPE'):
            return StrategyType(self.strategy_class.STRATEGY_TYPE)
            
        # Infer from class name
        class_name = self.strategy_class.__name__.lower()
        for stype in StrategyType:
            if stype.value in class_name:
                return stype
                
        return StrategyType.HYBRID
        
    async def optimize_with_walk_forward(
        self,
        window_size: int = 252,  # Trading days
        step_size: int = 63,     # Quarterly
        n_trials: int = 100
    ) -> WalkForwardResult:
        """Optimize strategy using walk-forward analysis.
        
        Args:
            window_size: Training window size in days
            step_size: Step forward size in days
            n_trials: Optimization trials per window
            
        Returns:
            Walk-forward optimization results
        """
        logger.info(f"Starting walk-forward optimization: window={window_size}, step={step_size}")
        
        in_sample_results = []
        out_of_sample_results = []
        parameter_history = []
        
        # Time series split
        n_splits = (len(self.data) - window_size) // step_size
        
        for i in range(n_splits):
            # Define train/test periods
            train_start = i * step_size
            train_end = train_start + window_size
            test_start = train_end
            test_end = min(test_start + step_size, len(self.data))
            
            train_data = self.data.iloc[train_start:train_end]
            test_data = self.data.iloc[test_start:test_end]
            
            logger.info(f"Walk-forward period {i+1}/{n_splits}")
            
            # Optimize on training data
            optimizer = self._create_optimizer(train_data)
            result = await optimizer.optimize(n_trials=n_trials)
            
            in_sample_results.append({
                'period': i,
                'params': result.best_params,
                'score': result.best_score,
                'convergence_iterations': result.n_iterations
            })
            
            parameter_history.append(result.best_params)
            
            # Evaluate on test data
            test_score = self._evaluate_out_of_sample(
                result.best_params,
                test_data
            )
            
            out_of_sample_results.append({
                'period': i,
                'params': result.best_params,
                'score': test_score,
                'data_points': len(test_data)
            })
            
        # Analyze results
        parameter_stability = self._calculate_parameter_stability(parameter_history)
        performance_consistency = self._calculate_performance_consistency(
            in_sample_results,
            out_of_sample_results
        )
        robust_parameters = self._select_robust_parameters(
            parameter_history,
            out_of_sample_results
        )
        sharpe_degradation = self._calculate_sharpe_degradation(
            in_sample_results,
            out_of_sample_results
        )
        
        return WalkForwardResult(
            in_sample_results=in_sample_results,
            out_of_sample_results=out_of_sample_results,
            parameter_stability=parameter_stability,
            performance_consistency=performance_consistency,
            robust_parameters=robust_parameters,
            sharpe_degradation=sharpe_degradation
        )
        
    async def create_ensemble_strategy(
        self,
        n_members: int = 5,
        diversity_threshold: float = 0.3,
        optimization_budget: int = 500
    ) -> EnsembleResult:
        """Create ensemble of diverse strategies.
        
        Args:
            n_members: Number of ensemble members
            diversity_threshold: Minimum correlation threshold
            optimization_budget: Total optimization trials
            
        Returns:
            Ensemble optimization results
        """
        logger.info(f"Creating ensemble with {n_members} members")
        
        # Allocate budget per member
        budget_per_member = optimization_budget // n_members
        
        # Optimize diverse strategies
        member_strategies = []
        member_performance = []
        
        for i in range(n_members):
            # Create variation of parameter space
            varied_space = self._create_parameter_variation(i, n_members)
            
            # Optimize with different initialization
            optimizer = OptimizationEngine(
                config=self.config,
                objective_function=self._create_objective_function(),
                parameter_space=varied_space
            )
            
            result = await optimizer.optimize(
                method='bayesian',
                n_trials=budget_per_member
            )
            
            member_strategies.append({
                'params': result.best_params,
                'score': result.best_score,
                'variation_id': i
            })
            
            member_performance.append(result.best_score)
            
        # Calculate strategy correlations
        correlation_matrix = self._calculate_strategy_correlations(member_strategies)
        
        # Select diverse subset if needed
        if self._needs_diversity_selection(correlation_matrix, diversity_threshold):
            selected_indices = self._select_diverse_strategies(
                correlation_matrix,
                member_performance,
                n_members
            )
            member_strategies = [member_strategies[i] for i in selected_indices]
            member_performance = [member_performance[i] for i in selected_indices]
            correlation_matrix = correlation_matrix[np.ix_(selected_indices, selected_indices)]
            
        # Optimize ensemble weights
        ensemble_weights = await self._optimize_ensemble_weights(
            member_strategies,
            member_performance
        )
        
        # Calculate ensemble performance
        ensemble_performance = self._calculate_ensemble_performance(
            member_strategies,
            ensemble_weights
        )
        
        # Calculate diversity score
        diversity_score = self._calculate_diversity_score(correlation_matrix)
        
        return EnsembleResult(
            member_strategies=member_strategies,
            ensemble_weights=ensemble_weights,
            individual_performance=member_performance,
            ensemble_performance=ensemble_performance,
            correlation_matrix=correlation_matrix,
            diversity_score=diversity_score
        )
        
    async def optimize_for_regime(
        self,
        market_regime: str,
        n_trials: int = 100
    ) -> OptimizationResult:
        """Optimize strategy for specific market regime.
        
        Args:
            market_regime: Market regime type
            n_trials: Number of optimization trials
            
        Returns:
            Regime-specific optimization results
        """
        # Filter data for regime
        regime_data = self._filter_data_by_regime(market_regime)
        
        # Adjust parameter space for regime
        regime_space = self._adjust_space_for_regime(market_regime)
        
        # Create regime-specific objective
        objective = self._create_regime_objective(market_regime)
        
        # Run optimization
        optimizer = OptimizationEngine(
            config=self.config,
            objective_function=objective,
            parameter_space=regime_space
        )
        
        result = await optimizer.optimize(
            method='bayesian',
            n_trials=n_trials
        )
        
        # Add regime metadata
        result.metadata['market_regime'] = market_regime
        result.metadata['regime_data_points'] = len(regime_data)
        
        return result
        
    def _create_optimizer(self, data: pd.DataFrame) -> OptimizationEngine:
        """Create optimizer instance for given data."""
        objective = self._create_objective_function(data)
        param_space = self._get_parameter_space()
        
        return OptimizationEngine(
            config=self.config,
            objective_function=objective,
            parameter_space=param_space
        )
        
    def _create_objective_function(
        self,
        data: Optional[pd.DataFrame] = None
    ) -> Callable:
        """Create objective function for optimization."""
        if data is None:
            data = self.data
            
        def objective(params: Dict[str, Any]) -> float:
            try:
                # Create strategy instance
                strategy = self.strategy_class(**params)
                
                # Evaluate performance
                metrics = self.evaluator.evaluate(strategy, data)
                
                # Composite score based on strategy type
                if self.strategy_type == StrategyType.MOMENTUM:
                    score = (
                        metrics.sharpe_ratio * 0.4 +
                        metrics.win_rate * 0.3 +
                        metrics.profit_factor * 0.3
                    )
                elif self.strategy_type == StrategyType.MEAN_REVERSION:
                    score = (
                        metrics.sharpe_ratio * 0.3 +
                        metrics.win_rate * 0.4 +
                        (1 / (1 + metrics.max_drawdown)) * 0.3
                    )
                else:
                    # Default scoring
                    score = metrics.sharpe_ratio
                    
                return float(score)
                
            except Exception as e:
                logger.error(f"Error in objective function: {str(e)}")
                return float('-inf')
                
        return objective
        
    def _get_parameter_space(self) -> Dict[str, Any]:
        """Get parameter space from strategy class."""
        if hasattr(self.strategy_class, 'get_parameter_space'):
            return self.strategy_class.get_parameter_space()
        else:
            # Default parameter space
            return {
                'lookback_period': (10, 200),
                'entry_threshold': (0.01, 0.1),
                'exit_threshold': (0.005, 0.05),
                'stop_loss': (0.01, 0.05),
                'take_profit': (0.02, 0.10)
            }
            
    def _evaluate_out_of_sample(
        self,
        params: Dict[str, Any],
        test_data: pd.DataFrame
    ) -> float:
        """Evaluate parameters on out-of-sample data."""
        strategy = self.strategy_class(**params)
        metrics = self.evaluator.evaluate(strategy, test_data)
        return metrics.sharpe_ratio
        
    def _calculate_parameter_stability(
        self,
        parameter_history: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate stability of parameters across periods."""
        stability_scores = {}
        
        for param_name in parameter_history[0].keys():
            values = [p[param_name] for p in parameter_history]
            
            if all(isinstance(v, (int, float)) for v in values):
                # Calculate coefficient of variation
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / mean_val if mean_val != 0 else float('inf')
                stability_scores[param_name] = 1 / (1 + cv)
            else:
                # For categorical, calculate mode frequency
                unique_values = len(set(values))
                stability_scores[param_name] = 1 / unique_values
                
        return stability_scores
        
    def _calculate_performance_consistency(
        self,
        in_sample: List[Dict[str, Any]],
        out_of_sample: List[Dict[str, Any]]
    ) -> float:
        """Calculate consistency between in-sample and out-of-sample."""
        in_scores = [r['score'] for r in in_sample]
        out_scores = [r['score'] for r in out_of_sample]
        
        # Correlation between in-sample and out-of-sample
        if len(in_scores) > 1:
            correlation = np.corrcoef(in_scores, out_scores)[0, 1]
            return max(0, correlation)
        else:
            return 0.0
            
    def _select_robust_parameters(
        self,
        parameter_history: List[Dict[str, Any]],
        performance_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Select most robust parameters."""
        # Weight recent periods more heavily
        weights = np.exp(np.linspace(-1, 0, len(parameter_history)))
        weights /= weights.sum()
        
        # Performance-weighted average for numeric parameters
        robust_params = {}
        scores = [p['score'] for p in performance_history]
        
        for param_name in parameter_history[0].keys():
            values = [p[param_name] for p in parameter_history]
            
            if all(isinstance(v, (int, float)) for v in values):
                # Weighted average
                weighted_value = np.average(values, weights=weights * scores)
                robust_params[param_name] = weighted_value
            else:
                # Most frequent value weighted by performance
                value_scores = {}
                for val, score in zip(values, scores):
                    if val not in value_scores:
                        value_scores[val] = 0
                    value_scores[val] += score
                    
                robust_params[param_name] = max(value_scores, key=value_scores.get)
                
        return robust_params
        
    def _calculate_sharpe_degradation(
        self,
        in_sample: List[Dict[str, Any]],
        out_of_sample: List[Dict[str, Any]]
    ) -> float:
        """Calculate Sharpe ratio degradation from in-sample to out-of-sample."""
        in_sharpes = [r['score'] for r in in_sample]
        out_sharpes = [r['score'] for r in out_of_sample]
        
        avg_in_sharpe = np.mean(in_sharpes)
        avg_out_sharpe = np.mean(out_sharpes)
        
        if avg_in_sharpe > 0:
            degradation = (avg_in_sharpe - avg_out_sharpe) / avg_in_sharpe
            return max(0, degradation)
        else:
            return 0.0
            
    def _create_parameter_variation(
        self,
        variation_id: int,
        total_variations: int
    ) -> Dict[str, Any]:
        """Create parameter space variation for ensemble diversity."""
        base_space = self._get_parameter_space()
        varied_space = {}
        
        # Create variations by shifting ranges
        shift_factor = 0.2 * (variation_id / total_variations - 0.5)
        
        for param_name, param_def in base_space.items():
            if isinstance(param_def, tuple) and len(param_def) == 2:
                # Shift continuous parameter range
                low, high = param_def
                range_size = high - low
                shift = shift_factor * range_size
                
                varied_space[param_name] = (
                    max(0, low + shift),
                    high + shift
                )
            else:
                # Keep categorical parameters same
                varied_space[param_name] = param_def
                
        return varied_space
        
    def _calculate_strategy_correlations(
        self,
        strategies: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Calculate correlation matrix between strategies."""
        n_strategies = len(strategies)
        returns_matrix = []
        
        # Simulate returns for each strategy
        for strategy in strategies:
            strat_instance = self.strategy_class(**strategy['params'])
            returns = self.evaluator.get_returns(strat_instance, self.data)
            returns_matrix.append(returns)
            
        returns_matrix = np.array(returns_matrix)
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(returns_matrix)
        
        return correlation_matrix
        
    def _needs_diversity_selection(
        self,
        correlation_matrix: np.ndarray,
        threshold: float
    ) -> bool:
        """Check if diversity selection is needed."""
        # Check if any pair has correlation above threshold
        n = correlation_matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if abs(correlation_matrix[i, j]) > threshold:
                    return True
        return False
        
    def _select_diverse_strategies(
        self,
        correlation_matrix: np.ndarray,
        performance: List[float],
        n_select: int
    ) -> List[int]:
        """Select diverse strategies using clustering."""
        from sklearn.cluster import AgglomerativeClustering
        
        # Use correlation distance
        distance_matrix = 1 - np.abs(correlation_matrix)
        
        # Hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_select,
            metric='precomputed',
            linkage='average'
        )
        
        clusters = clustering.fit_predict(distance_matrix)
        
        # Select best from each cluster
        selected = []
        for cluster_id in range(n_select):
            cluster_indices = np.where(clusters == cluster_id)[0]
            cluster_performance = [performance[i] for i in cluster_indices]
            best_idx = cluster_indices[np.argmax(cluster_performance)]
            selected.append(best_idx)
            
        return selected
        
    async def _optimize_ensemble_weights(
        self,
        strategies: List[Dict[str, Any]],
        performance: List[float]
    ) -> List[float]:
        """Optimize ensemble weights for maximum Sharpe."""
        n_strategies = len(strategies)
        
        # Get returns for each strategy
        returns_matrix = []
        for strategy in strategies:
            strat_instance = self.strategy_class(**strategy['params'])
            returns = self.evaluator.get_returns(strat_instance, self.data)
            returns_matrix.append(returns)
            
        returns_matrix = np.array(returns_matrix).T
        
        # Optimize weights using mean-variance optimization
        from scipy.optimize import minimize
        
        def negative_sharpe(weights):
            portfolio_returns = returns_matrix @ weights
            return -np.mean(portfolio_returns) / np.std(portfolio_returns)
            
        # Constraints: weights sum to 1, all non-negative
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: w}
        ]
        
        # Initial guess: equal weights
        initial_weights = np.ones(n_strategies) / n_strategies
        
        # Optimize
        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            constraints=constraints
        )
        
        return result.x.tolist()
        
    def _calculate_ensemble_performance(
        self,
        strategies: List[Dict[str, Any]],
        weights: List[float]
    ) -> float:
        """Calculate ensemble performance."""
        # Get weighted returns
        weighted_returns = 0
        
        for strategy, weight in zip(strategies, weights):
            strat_instance = self.strategy_class(**strategy['params'])
            metrics = self.evaluator.evaluate(strat_instance, self.data)
            weighted_returns += weight * metrics.sharpe_ratio
            
        return weighted_returns
        
    def _calculate_diversity_score(self, correlation_matrix: np.ndarray) -> float:
        """Calculate diversity score from correlation matrix."""
        n = correlation_matrix.shape[0]
        
        # Average absolute correlation (excluding diagonal)
        total_corr = 0
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                total_corr += abs(correlation_matrix[i, j])
                count += 1
                
        avg_corr = total_corr / count if count > 0 else 0
        
        # Diversity score: 1 - average correlation
        return 1 - avg_corr
        
    def _filter_data_by_regime(self, regime: str) -> pd.DataFrame:
        """Filter data for specific market regime."""
        # Implement regime detection logic
        # This is a placeholder - would need actual regime detection
        return self.data
        
    def _adjust_space_for_regime(self, regime: str) -> Dict[str, Any]:
        """Adjust parameter space for market regime."""
        base_space = self._get_parameter_space()
        
        # Regime-specific adjustments
        if regime == 'high_volatility':
            # Wider stops, shorter lookbacks
            if 'stop_loss' in base_space:
                base_space['stop_loss'] = (0.02, 0.10)
            if 'lookback_period' in base_space:
                base_space['lookback_period'] = (5, 50)
        elif regime == 'trending':
            # Tighter stops, longer lookbacks
            if 'stop_loss' in base_space:
                base_space['stop_loss'] = (0.005, 0.03)
            if 'lookback_period' in base_space:
                base_space['lookback_period'] = (20, 200)
                
        return base_space
        
    def _create_regime_objective(self, regime: str) -> Callable:
        """Create regime-specific objective function."""
        base_objective = self._create_objective_function()
        
        def regime_objective(params: Dict[str, Any]) -> float:
            score = base_objective(params)
            
            # Apply regime-specific adjustments
            if regime == 'high_volatility':
                # Penalize strategies with poor drawdown control
                strategy = self.strategy_class(**params)
                metrics = self.evaluator.evaluate(strategy, self.data)
                score *= (1 - metrics.max_drawdown)
            elif regime == 'sideways':
                # Favor mean-reversion characteristics
                strategy = self.strategy_class(**params)
                metrics = self.evaluator.evaluate(strategy, self.data)
                score *= metrics.win_rate
                
            return score
            
        return regime_objective


class StrategyEvaluator:
    """Evaluates trading strategy performance."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize evaluator with market data."""
        self.data = data
        
    def evaluate(self, strategy: Any, data: Optional[pd.DataFrame] = None) -> Any:
        """Evaluate strategy performance."""
        if data is None:
            data = self.data
            
        # Run backtest
        returns = self.get_returns(strategy, data)
        
        # Calculate metrics
        metrics = self._calculate_metrics(returns)
        
        return metrics
        
    def get_returns(self, strategy: Any, data: pd.DataFrame) -> np.ndarray:
        """Get strategy returns."""
        # Placeholder - implement actual backtesting logic
        # This would integrate with your backtesting engine
        n_days = len(data)
        returns = np.random.normal(0.0002, 0.01, n_days)
        
        return returns
        
    def _calculate_metrics(self, returns: np.ndarray) -> Any:
        """Calculate performance metrics."""
        from dataclasses import dataclass
        
        @dataclass
        class PerformanceMetrics:
            sharpe_ratio: float
            calmar_ratio: float
            win_rate: float
            profit_factor: float
            max_drawdown: float
            
        # Calculate metrics
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        # Calculate drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(np.min(drawdown))
        
        # Other metrics
        win_rate = np.sum(returns > 0) / len(returns)
        winning_returns = returns[returns > 0]
        losing_returns = abs(returns[returns < 0])
        
        profit_factor = (
            np.sum(winning_returns) / np.sum(losing_returns)
            if np.sum(losing_returns) > 0 else np.inf
        )
        
        annual_return = np.mean(returns) * 252
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        return PerformanceMetrics(
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown
        )