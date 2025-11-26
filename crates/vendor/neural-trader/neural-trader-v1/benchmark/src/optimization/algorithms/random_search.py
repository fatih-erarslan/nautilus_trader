"""Random Search implementation for parameter optimization."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
import logging
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import time
import random

logger = logging.getLogger(__name__)


@dataclass
class RandomSearchResult:
    """Result from random search optimization."""
    best_params: Dict[str, Any]
    best_score: float
    all_results: pd.DataFrame
    search_time: float
    total_evaluations: int
    convergence_curve: List[float]


class RandomSearch:
    """Random search optimizer."""
    
    def __init__(
        self,
        objective_function: Callable,
        parameter_space: Dict[str, Any],
        n_jobs: int = -1,
        random_state: Optional[int] = None
    ):
        """Initialize random search.
        
        Args:
            objective_function: Function to optimize
            parameter_space: Dictionary defining parameter bounds
            n_jobs: Number of parallel jobs
            random_state: Random seed
        """
        self.objective_function = objective_function
        self.parameter_space = parameter_space
        self.n_jobs = n_jobs
        
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
            
        self.results = []
        self.best_score = float('-inf')
        self.best_params = None
        
    def _sample_parameters(self) -> Dict[str, Any]:
        """Sample random parameters."""
        params = {}
        
        for param_name, param_def in self.parameter_space.items():
            if isinstance(param_def, tuple) and len(param_def) == 2:
                # Continuous parameter
                low, high = param_def
                params[param_name] = np.random.uniform(low, high)
            elif isinstance(param_def, list):
                # Categorical parameter
                params[param_name] = random.choice(param_def)
            elif isinstance(param_def, dict):
                # Detailed specification
                if param_def.get('type') == 'integer':
                    low, high = param_def['low'], param_def['high']
                    params[param_name] = np.random.randint(low, high + 1)
                elif param_def.get('type') == 'log-uniform':
                    low, high = param_def['low'], param_def['high']
                    log_low, log_high = np.log(low), np.log(high)
                    log_val = np.random.uniform(log_low, log_high)
                    params[param_name] = np.exp(log_val)
                else:
                    # Default to uniform
                    low, high = param_def['low'], param_def['high']
                    params[param_name] = np.random.uniform(low, high)
            else:
                # Single value
                params[param_name] = param_def
                
        return params
        
    def optimize(
        self,
        n_iter: int = 100,
        callback: Optional[Callable] = None
    ) -> RandomSearchResult:
        """Run random search optimization."""
        start_time = time.time()
        
        logger.info(f"Random search with {n_iter} iterations")
        
        # Generate random parameter sets
        param_sets = [self._sample_parameters() for _ in range(n_iter)]
        
        # Evaluate parameters
        if self.n_jobs == 1:
            # Sequential evaluation
            for i, params in enumerate(param_sets):
                score = self._evaluate_params(params)
                result = {
                    'params': params,
                    'score': score,
                    'iteration': i
                }
                self.results.append(result)
                
                # Update best
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                    
                if callback:
                    callback(result)
        else:
            # Parallel evaluation
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = []
                for i, params in enumerate(param_sets):
                    future = executor.submit(self._evaluate_params, params)
                    futures.append((i, params, future))
                    
                for i, params, future in futures:
                    score = future.result()
                    result = {
                        'params': params,
                        'score': score,
                        'iteration': i
                    }
                    self.results.append(result)
                    
                    # Update best
                    if score > self.best_score:
                        self.best_score = score
                        self.best_params = params.copy()
                        
                    if callback:
                        callback(result)
                        
        # Sort results by iteration
        self.results.sort(key=lambda x: x['iteration'])
        
        # Create convergence curve
        convergence_curve = []
        current_best = float('-inf')
        for result in self.results:
            current_best = max(current_best, result['score'])
            convergence_curve.append(current_best)
            
        # Create results DataFrame
        results_df = pd.DataFrame(self.results)
        
        search_time = time.time() - start_time
        
        return RandomSearchResult(
            best_params=self.best_params,
            best_score=self.best_score,
            all_results=results_df,
            search_time=search_time,
            total_evaluations=len(self.results),
            convergence_curve=convergence_curve
        )
        
    def _evaluate_params(self, params: Dict[str, Any]) -> float:
        """Evaluate parameters."""
        try:
            return float(self.objective_function(params))
        except Exception as e:
            logger.error(f"Error evaluating {params}: {str(e)}")
            return float('-inf')