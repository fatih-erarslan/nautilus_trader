"""Win rate optimization objective."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class WinRateObjective:
    """Win rate optimization objective."""
    
    def __init__(
        self,
        strategy_evaluator: Callable,
        data: pd.DataFrame,
        threshold: float = 0.0,
        min_trades: int = 10
    ):
        """Initialize win rate objective.
        
        Args:
            strategy_evaluator: Function that evaluates strategy
            data: Market data
            threshold: Threshold for winning trades
            min_trades: Minimum number of trades required
        """
        self.strategy_evaluator = strategy_evaluator
        self.data = data
        self.threshold = threshold
        self.min_trades = min_trades
        
    def __call__(self, params: Dict[str, Any]) -> float:
        """Compute win rate."""
        try:
            returns = self.strategy_evaluator(params, self.data)
            
            if returns is None or len(returns) < self.min_trades:
                return 0.0
                
            win_rate = np.mean(returns > self.threshold)
            return win_rate
            
        except Exception as e:
            logger.error(f"Error calculating win rate: {str(e)}")
            return 0.0


class ProfitFactorObjective:
    """Profit factor optimization objective."""
    
    def __init__(
        self,
        strategy_evaluator: Callable,
        data: pd.DataFrame
    ):
        """Initialize profit factor objective."""
        self.strategy_evaluator = strategy_evaluator
        self.data = data
        
    def __call__(self, params: Dict[str, Any]) -> float:
        """Compute profit factor."""
        try:
            returns = self.strategy_evaluator(params, self.data)
            
            if returns is None or len(returns) == 0:
                return 0.0
                
            winning_returns = returns[returns > 0]
            losing_returns = returns[returns < 0]
            
            if len(losing_returns) == 0:
                return float('inf') if len(winning_returns) > 0 else 0.0
                
            profit_factor = np.sum(winning_returns) / abs(np.sum(losing_returns))
            return profit_factor
            
        except Exception as e:
            logger.error(f"Error calculating profit factor: {str(e)}")
            return 0.0