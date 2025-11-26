"""Sharpe ratio optimization objective for risk-adjusted returns."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SharpeMetrics:
    """Sharpe ratio calculation metrics."""
    sharpe_ratio: float
    annual_return: float
    annual_volatility: float
    excess_return: float
    risk_free_rate: float
    n_observations: int


class SharpeRatioObjective:
    """Sharpe ratio optimization objective function."""
    
    def __init__(
        self,
        strategy_evaluator: Callable,
        data: pd.DataFrame,
        risk_free_rate: float = 0.02,
        annualization_factor: float = 252.0,
        min_observations: int = 30,
        volatility_floor: float = 1e-6,
        return_penalty: float = 0.0
    ):
        """Initialize Sharpe ratio objective.
        
        Args:
            strategy_evaluator: Function that evaluates strategy given parameters
            data: Market data for backtesting
            risk_free_rate: Annual risk-free rate
            annualization_factor: Factor to annualize returns (252 for daily)
            min_observations: Minimum observations required
            volatility_floor: Minimum volatility to avoid division by zero
            return_penalty: Penalty for negative returns
        """
        self.strategy_evaluator = strategy_evaluator
        self.data = data
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor
        self.min_observations = min_observations
        self.volatility_floor = volatility_floor
        self.return_penalty = return_penalty
        
        # Cache for computed results
        self._cache = {}
        
    def __call__(self, params: Dict[str, Any]) -> float:
        """Compute Sharpe ratio for given parameters.
        
        Args:
            params: Strategy parameters
            
        Returns:
            Sharpe ratio value
        """
        # Check cache
        param_key = self._params_to_key(params)
        if param_key in self._cache:
            return self._cache[param_key]
            
        try:
            # Evaluate strategy
            strategy_returns = self.strategy_evaluator(params, self.data)
            
            if strategy_returns is None or len(strategy_returns) < self.min_observations:
                logger.warning(f"Insufficient observations: {len(strategy_returns) if strategy_returns is not None else 0}")
                return float('-inf')
                
            # Calculate Sharpe ratio
            metrics = self.calculate_sharpe_metrics(strategy_returns)
            sharpe_ratio = metrics.sharpe_ratio
            
            # Apply penalties
            if metrics.annual_return < 0 and self.return_penalty > 0:
                sharpe_ratio -= self.return_penalty * abs(metrics.annual_return)
                
            # Cache result
            self._cache[param_key] = sharpe_ratio
            
            return sharpe_ratio
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio for {params}: {str(e)}")
            return float('-inf')
            
    def calculate_sharpe_metrics(self, returns: np.ndarray) -> SharpeMetrics:
        """Calculate detailed Sharpe ratio metrics.
        
        Args:
            returns: Array of strategy returns
            
        Returns:
            SharpeMetrics with detailed calculations
        """
        if len(returns) == 0:
            return SharpeMetrics(
                sharpe_ratio=float('-inf'),
                annual_return=0.0,
                annual_volatility=0.0,
                excess_return=0.0,
                risk_free_rate=self.risk_free_rate,
                n_observations=0
            )
            
        # Convert to numpy array
        returns = np.asarray(returns)
        
        # Remove NaN values
        returns = returns[~np.isnan(returns)]
        
        if len(returns) < self.min_observations:
            return SharpeMetrics(
                sharpe_ratio=float('-inf'),
                annual_return=0.0,
                annual_volatility=0.0,
                excess_return=0.0,
                risk_free_rate=self.risk_free_rate,
                n_observations=len(returns)
            )
            
        # Calculate basic statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # Annualize
        annual_return = mean_return * self.annualization_factor
        annual_volatility = std_return * np.sqrt(self.annualization_factor)
        
        # Apply volatility floor
        annual_volatility = max(annual_volatility, self.volatility_floor)
        
        # Calculate excess return
        excess_return = annual_return - self.risk_free_rate
        
        # Calculate Sharpe ratio
        sharpe_ratio = excess_return / annual_volatility
        
        return SharpeMetrics(
            sharpe_ratio=sharpe_ratio,
            annual_return=annual_return,
            annual_volatility=annual_volatility,
            excess_return=excess_return,
            risk_free_rate=self.risk_free_rate,
            n_observations=len(returns)
        )
        
    def calculate_rolling_sharpe(
        self,
        returns: np.ndarray,
        window: int = 252
    ) -> np.ndarray:
        """Calculate rolling Sharpe ratio.
        
        Args:
            returns: Array of strategy returns
            window: Rolling window size
            
        Returns:
            Array of rolling Sharpe ratios
        """
        if len(returns) < window:
            return np.array([])
            
        rolling_sharpe = []
        
        for i in range(window, len(returns) + 1):
            window_returns = returns[i - window:i]
            metrics = self.calculate_sharpe_metrics(window_returns)
            rolling_sharpe.append(metrics.sharpe_ratio)
            
        return np.array(rolling_sharpe)
        
    def calculate_information_ratio(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> float:
        """Calculate information ratio vs benchmark.
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Information ratio
        """
        if len(returns) != len(benchmark_returns) or len(returns) == 0:
            return float('-inf')
            
        # Active returns
        active_returns = returns - benchmark_returns
        
        # Active return and tracking error
        active_return = np.mean(active_returns) * self.annualization_factor
        tracking_error = np.std(active_returns, ddof=1) * np.sqrt(self.annualization_factor)
        
        if tracking_error == 0:
            return float('inf') if active_return > 0 else float('-inf')
            
        return active_return / tracking_error
        
    def calculate_conditional_sharpe(
        self,
        returns: np.ndarray,
        percentile: float = 0.05
    ) -> float:
        """Calculate conditional Sharpe ratio (Sharpe of worst returns).
        
        Args:
            returns: Strategy returns
            percentile: Percentile threshold for worst returns
            
        Returns:
            Conditional Sharpe ratio
        """
        if len(returns) == 0:
            return float('-inf')
            
        # Get worst returns
        threshold = np.percentile(returns, percentile * 100)
        worst_returns = returns[returns <= threshold]
        
        if len(worst_returns) == 0:
            return float('-inf')
            
        # Calculate Sharpe for worst returns
        metrics = self.calculate_sharpe_metrics(worst_returns)
        return metrics.sharpe_ratio
        
    def calculate_adjusted_sharpe(
        self,
        returns: np.ndarray,
        skewness_penalty: float = 1.0,
        kurtosis_penalty: float = 1.0
    ) -> float:
        """Calculate skewness and kurtosis adjusted Sharpe ratio.
        
        Args:
            returns: Strategy returns
            skewness_penalty: Penalty factor for negative skewness
            kurtosis_penalty: Penalty factor for excess kurtosis
            
        Returns:
            Adjusted Sharpe ratio
        """
        from scipy import stats
        
        if len(returns) < 4:
            return float('-inf')
            
        # Basic Sharpe ratio
        metrics = self.calculate_sharpe_metrics(returns)
        sharpe = metrics.sharpe_ratio
        
        # Calculate higher moments
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns, fisher=True)  # Excess kurtosis
        
        # Adjust for skewness (penalize negative skewness)
        if skewness < 0:
            sharpe_adjusted = sharpe * (1 + skewness_penalty * abs(skewness) / 6)
        else:
            sharpe_adjusted = sharpe
            
        # Adjust for kurtosis (penalize excess kurtosis)
        if kurtosis > 0:
            sharpe_adjusted = sharpe_adjusted * (1 - kurtosis_penalty * kurtosis / 24)
            
        return sharpe_adjusted
        
    def _params_to_key(self, params: Dict[str, Any]) -> str:
        """Convert parameters to cache key."""
        import json
        return json.dumps(params, sort_keys=True)
        
    def clear_cache(self):
        """Clear computation cache."""
        self._cache.clear()
        
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'cache_size': len(self._cache),
            'cache_hits': getattr(self, '_cache_hits', 0),
            'cache_misses': getattr(self, '_cache_misses', 0)
        }


class ModifiedSharpeObjective(SharpeRatioObjective):
    """Modified Sharpe ratio with drawdown consideration."""
    
    def __init__(
        self,
        strategy_evaluator: Callable,
        data: pd.DataFrame,
        drawdown_weight: float = 0.5,
        **kwargs
    ):
        """Initialize modified Sharpe objective.
        
        Args:
            strategy_evaluator: Function that evaluates strategy
            data: Market data
            drawdown_weight: Weight for drawdown penalty
            **kwargs: Additional arguments for base class
        """
        super().__init__(strategy_evaluator, data, **kwargs)
        self.drawdown_weight = drawdown_weight
        
    def __call__(self, params: Dict[str, Any]) -> float:
        """Compute modified Sharpe ratio."""
        try:
            # Get basic Sharpe ratio
            sharpe_ratio = super().__call__(params)
            
            if np.isinf(sharpe_ratio):
                return sharpe_ratio
                
            # Evaluate strategy for drawdown
            strategy_returns = self.strategy_evaluator(params, self.data)
            
            if strategy_returns is None or len(strategy_returns) == 0:
                return float('-inf')
                
            # Calculate maximum drawdown
            cumulative_returns = np.cumprod(1 + strategy_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(np.min(drawdown))
            
            # Modify Sharpe ratio with drawdown penalty
            if max_drawdown > 0:
                modified_sharpe = sharpe_ratio * (1 - self.drawdown_weight * max_drawdown)
            else:
                modified_sharpe = sharpe_ratio
                
            return modified_sharpe
            
        except Exception as e:
            logger.error(f"Error calculating modified Sharpe ratio: {str(e)}")
            return float('-inf')


class ProbabilisticSharpeObjective(SharpeRatioObjective):
    """Probabilistic Sharpe ratio considering estimation uncertainty."""
    
    def __init__(
        self,
        strategy_evaluator: Callable,
        data: pd.DataFrame,
        benchmark_sharpe: float = 0.0,
        confidence_level: float = 0.95,
        **kwargs
    ):
        """Initialize probabilistic Sharpe objective.
        
        Args:
            strategy_evaluator: Function that evaluates strategy
            data: Market data
            benchmark_sharpe: Benchmark Sharpe ratio
            confidence_level: Confidence level for PSR
            **kwargs: Additional arguments for base class
        """
        super().__init__(strategy_evaluator, data, **kwargs)
        self.benchmark_sharpe = benchmark_sharpe
        self.confidence_level = confidence_level
        
    def __call__(self, params: Dict[str, Any]) -> float:
        """Compute probabilistic Sharpe ratio."""
        from scipy import stats
        
        try:
            # Get strategy returns
            strategy_returns = self.strategy_evaluator(params, self.data)
            
            if strategy_returns is None or len(strategy_returns) < self.min_observations:
                return float('-inf')
                
            # Calculate basic Sharpe metrics
            metrics = self.calculate_sharpe_metrics(strategy_returns)
            
            if np.isinf(metrics.sharpe_ratio):
                return float('-inf')
                
            # Calculate skewness and kurtosis
            skewness = stats.skew(strategy_returns)
            kurtosis = stats.kurtosis(strategy_returns, fisher=True)
            
            # Calculate PSR
            n = len(strategy_returns)
            
            # Standard error of Sharpe ratio
            se_sharpe = np.sqrt((1 + 0.5 * metrics.sharpe_ratio**2 - 
                               skewness * metrics.sharpe_ratio + 
                               (kurtosis - 3) / 4 * metrics.sharpe_ratio**2) / (n - 1))
            
            # Z-score for PSR
            z_score = (metrics.sharpe_ratio - self.benchmark_sharpe) / se_sharpe
            
            # Probabilistic Sharpe ratio
            psr = stats.norm.cdf(z_score)
            
            # Convert to optimization objective (higher is better)
            return psr * metrics.sharpe_ratio
            
        except Exception as e:
            logger.error(f"Error calculating PSR: {str(e)}")
            return float('-inf')