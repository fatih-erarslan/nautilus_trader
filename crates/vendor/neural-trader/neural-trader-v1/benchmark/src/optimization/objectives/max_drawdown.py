"""Maximum drawdown optimization objective."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class MaxDrawdownObjective:
    """Maximum drawdown minimization objective."""
    
    def __init__(
        self,
        strategy_evaluator: Callable,
        data: pd.DataFrame,
        minimize: bool = True,
        return_weight: float = 0.5
    ):
        """Initialize max drawdown objective.
        
        Args:
            strategy_evaluator: Function that evaluates strategy
            data: Market data
            minimize: Whether to minimize drawdown (True) or maximize negative drawdown
            return_weight: Weight for return component in combined objective
        """
        self.strategy_evaluator = strategy_evaluator
        self.data = data
        self.minimize = minimize
        self.return_weight = return_weight
        
    def __call__(self, params: Dict[str, Any]) -> float:
        """Compute drawdown objective."""
        try:
            returns = self.strategy_evaluator(params, self.data)
            
            if returns is None or len(returns) == 0:
                return float('-inf') if not self.minimize else float('inf')
                
            # Calculate maximum drawdown
            max_dd = self.calculate_max_drawdown(returns)
            
            if self.minimize:
                # For minimization, return negative drawdown (higher is better)
                return -max_dd
            else:
                # For maximization, penalize high drawdown
                total_return = np.sum(returns)
                return self.return_weight * total_return - (1 - self.return_weight) * max_dd
                
        except Exception as e:
            logger.error(f"Error calculating drawdown: {str(e)}")
            return float('-inf') if not self.minimize else float('inf')
            
    def calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
            
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return abs(np.min(drawdown))


class WinRateObjective:
    """Win rate optimization objective."""
    
    def __init__(
        self,
        strategy_evaluator: Callable,
        data: pd.DataFrame,
        threshold: float = 0.0
    ):
        """Initialize win rate objective.
        
        Args:
            strategy_evaluator: Function that evaluates strategy
            data: Market data
            threshold: Threshold for winning trades
        """
        self.strategy_evaluator = strategy_evaluator
        self.data = data
        self.threshold = threshold
        
    def __call__(self, params: Dict[str, Any]) -> float:
        """Compute win rate."""
        try:
            returns = self.strategy_evaluator(params, self.data)
            
            if returns is None or len(returns) == 0:
                return 0.0
                
            win_rate = np.mean(returns > self.threshold)
            return win_rate
            
        except Exception as e:
            logger.error(f"Error calculating win rate: {str(e)}")
            return 0.0