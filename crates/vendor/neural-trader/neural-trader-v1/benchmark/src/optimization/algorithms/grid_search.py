"""Grid Search implementation for exhaustive parameter exploration."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Tuple
import logging
from dataclasses import dataclass
from itertools import product
from concurrent.futures import ProcessPoolExecutor
import time

logger = logging.getLogger(__name__)


@dataclass
class GridSearchResult:
    """Result from grid search optimization."""
    best_params: Dict[str, Any]
    best_score: float
    all_results: pd.DataFrame
    search_time: float
    total_evaluations: int
    grid_coverage: float


class GridSearch:
    """Exhaustive grid search optimizer."""
    
    def __init__(
        self,
        objective_function: Callable,
        parameter_space: Dict[str, Any],
        n_jobs: int = -1
    ):
        """Initialize grid search.
        
        Args:
            objective_function: Function to optimize
            parameter_space: Dictionary defining parameter grids
            n_jobs: Number of parallel jobs
        """
        self.objective_function = objective_function
        self.parameter_space = parameter_space
        self.n_jobs = n_jobs
        
        self.results = []
        
    def _create_parameter_grid(
        self,
        resolution: Optional[Dict[str, int]] = None
    ) -> List[Dict[str, Any]]:
        """Create parameter grid."""
        if resolution is None:
            resolution = {}
            
        param_values = {}
        
        for param_name, param_def in self.parameter_space.items():
            if isinstance(param_def, tuple) and len(param_def) == 2:
                # Continuous parameter
                n_points = resolution.get(param_name, 10)
                values = np.linspace(param_def[0], param_def[1], n_points)
                param_values[param_name] = values
            elif isinstance(param_def, list):
                # Categorical parameter
                param_values[param_name] = param_def
            else:
                # Single value
                param_values[param_name] = [param_def]
                
        # Generate all combinations
        param_names = list(param_values.keys())
        param_combinations = list(product(*param_values.values()))
        
        grid = []
        for combination in param_combinations:
            params = dict(zip(param_names, combination))
            grid.append(params)
            
        return grid
        
    def optimize(
        self,
        resolution: Optional[Dict[str, int]] = None,
        callback: Optional[Callable] = None
    ) -> GridSearchResult:
        """Run grid search optimization."""
        start_time = time.time()
        
        # Create parameter grid
        grid = self._create_parameter_grid(resolution)
        logger.info(f"Grid search with {len(grid)} parameter combinations")
        
        # Evaluate all combinations
        if self.n_jobs == 1:
            # Sequential evaluation
            for i, params in enumerate(grid):
                score = self._evaluate_params(params)
                result = {
                    'params': params,
                    'score': score,
                    'evaluation': i
                }
                self.results.append(result)
                
                if callback:
                    callback(result)
        else:
            # Parallel evaluation
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = []
                for i, params in enumerate(grid):
                    future = executor.submit(self._evaluate_params, params)
                    futures.append((i, params, future))
                    
                for i, params, future in futures:
                    score = future.result()
                    result = {
                        'params': params,
                        'score': score,
                        'evaluation': i
                    }
                    self.results.append(result)
                    
                    if callback:
                        callback(result)
                        
        # Find best result
        best_result = max(self.results, key=lambda x: x['score'])
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results)
        
        search_time = time.time() - start_time
        
        return GridSearchResult(
            best_params=best_result['params'],
            best_score=best_result['score'],
            all_results=results_df,
            search_time=search_time,
            total_evaluations=len(self.results),
            grid_coverage=1.0  # Full coverage for grid search
        )
        
    def _evaluate_params(self, params: Dict[str, Any]) -> float:
        """Evaluate parameters."""
        try:
            return float(self.objective_function(params))
        except Exception as e:
            logger.error(f"Error evaluating {params}: {str(e)}")
            return float('-inf')