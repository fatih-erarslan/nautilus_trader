"""
Parameter optimization algorithms for trading system optimization.

Provides grid search, random search, and smart search strategies
for optimizing trading system parameters.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Callable, Optional, Union
from dataclasses import dataclass, field
import time
import itertools
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    best_params: Dict[str, Any]
    best_score: float
    history: List[Dict[str, Any]]
    converged: bool
    iterations: int
    metrics: Dict[str, Any] = field(default_factory=dict)
    convergence_iteration: Optional[int] = None
    early_stopped: bool = False
    time_stopped: bool = False
    callback_stopped: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Export results to dictionary."""
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'history': self.history,
            'converged': self.converged,
            'iterations': self.iterations,
            'metrics': self.metrics,
            'convergence_iteration': self.convergence_iteration,
            'early_stopped': self.early_stopped
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Optimization Results",
            "=" * 50,
            f"Best Score: {self.best_score:.6f}",
            f"Converged: {self.converged}",
            f"Iterations: {self.iterations}",
            "",
            "Best Parameters:",
        ]
        
        for param, value in self.best_params.items():
            if isinstance(value, float):
                lines.append(f"  {param}: {value:.6f}")
            else:
                lines.append(f"  {param}: {value}")
        
        if self.metrics:
            lines.extend(["", "Metrics:"])
            for metric, value in self.metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {metric}: {value:.6f}")
                elif isinstance(value, list):
                    lines.append(f"  {metric}: {len(value)} items")
                else:
                    lines.append(f"  {metric}: {value}")
        
        return "\n".join(lines)


class BaseOptimizer(ABC):
    """Base class for parameter optimizers."""
    
    def __init__(self, search_space: Dict[str, Dict[str, Any]], 
                 constraints: Optional[List[Callable]] = None):
        """
        Initialize optimizer.
        
        Args:
            search_space: Parameter definitions
            constraints: List of constraint functions
        """
        self.search_space = search_space
        self.constraints = constraints or []
        self._validate_search_space()
    
    def _validate_search_space(self):
        """Validate search space definition."""
        for param, config in self.search_space.items():
            if 'type' not in config:
                raise ValueError(f"Parameter {param} missing 'type'")
            
            param_type = config['type']
            if param_type in ['float', 'int']:
                if 'min' not in config or 'max' not in config:
                    raise ValueError(f"Parameter {param} missing 'min' or 'max'")
            elif param_type == 'categorical':
                if 'values' not in config:
                    raise ValueError(f"Parameter {param} missing 'values'")
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate parameters against search space and constraints.
        
        Args:
            params: Parameters to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check all required parameters present
        if set(params.keys()) != set(self.search_space.keys()):
            return False
        
        # Check parameter bounds
        for param, value in params.items():
            config = self.search_space[param]
            param_type = config['type']
            
            if param_type == 'float' or param_type == 'int':
                if value < config['min'] or value > config['max']:
                    return False
                if param_type == 'int' and not isinstance(value, int):
                    return False
            elif param_type == 'categorical':
                if value not in config['values']:
                    return False
        
        # Check constraints
        for constraint in self.constraints:
            if not constraint(params):
                return False
        
        return True
    
    def sample_params(self) -> Dict[str, Any]:
        """Sample random parameters from search space."""
        params = {}
        
        for param, config in self.search_space.items():
            param_type = config['type']
            
            if param_type == 'float':
                params[param] = np.random.uniform(config['min'], config['max'])
            elif param_type == 'int':
                params[param] = np.random.randint(config['min'], config['max'] + 1)
            elif param_type == 'categorical':
                params[param] = np.random.choice(config['values'])
        
        return params
    
    @abstractmethod
    def optimize(self, objective_function: Callable, **kwargs) -> OptimizationResult:
        """
        Run optimization.
        
        Args:
            objective_function: Function to minimize
            **kwargs: Additional optimization parameters
            
        Returns:
            OptimizationResult with best parameters and history
        """
        pass
    
    def _check_convergence(self, history: List[Dict], tolerance: float = 1e-6,
                          window: int = 10, relative: bool = False) -> bool:
        """Check if optimization has converged."""
        if len(history) < window:
            return False
        
        recent_scores = [h['score'] for h in history[-window:]]
        
        if relative:
            # Relative change
            if abs(recent_scores[0]) < 1e-10:
                return True
            change = abs(recent_scores[-1] - recent_scores[0]) / abs(recent_scores[0])
            return change < tolerance
        else:
            # Absolute change
            return max(recent_scores) - min(recent_scores) < tolerance


class GridSearchOptimizer(BaseOptimizer):
    """Grid search parameter optimization."""
    
    def __init__(self, search_space: Dict[str, Dict[str, Any]], 
                 resolution: Dict[str, int],
                 constraints: Optional[List[Callable]] = None):
        """
        Initialize grid search optimizer.
        
        Args:
            search_space: Parameter definitions
            resolution: Number of grid points per parameter
            constraints: List of constraint functions
        """
        super().__init__(search_space, constraints)
        self.resolution = resolution
        self.total_evaluations = self._calculate_total_evaluations()
    
    def _calculate_total_evaluations(self) -> int:
        """Calculate total number of grid evaluations."""
        total = 1
        for param, res in self.resolution.items():
            total *= res
        return total
    
    def _generate_grid(self) -> List[Dict[str, Any]]:
        """Generate parameter grid."""
        param_ranges = {}
        
        for param, config in self.search_space.items():
            param_type = config['type']
            res = self.resolution.get(param, 10)
            
            if param_type == 'float':
                param_ranges[param] = np.linspace(config['min'], config['max'], res)
            elif param_type == 'int':
                param_ranges[param] = np.linspace(
                    config['min'], config['max'], res
                ).astype(int)
            elif param_type == 'categorical':
                param_ranges[param] = config['values']
        
        # Generate all combinations
        param_names = list(param_ranges.keys())
        param_values = [param_ranges[name] for name in param_names]
        
        grid = []
        for values in itertools.product(*param_values):
            params = dict(zip(param_names, values))
            if self.validate_params(params):
                grid.append(params)
        
        return grid
    
    def optimize(self, objective_function: Callable,
                max_evaluations: Optional[int] = None,
                target_score: Optional[float] = None,
                convergence_tolerance: Optional[float] = None,
                convergence_window: int = 10,
                parallel: bool = False,
                num_workers: int = 4) -> OptimizationResult:
        """
        Run grid search optimization.
        
        Args:
            objective_function: Function to minimize
            max_evaluations: Maximum evaluations (default: all grid points)
            target_score: Stop if score below target
            convergence_tolerance: Stop if converged
            convergence_window: Window for convergence check
            parallel: Use parallel evaluation
            num_workers: Number of parallel workers
            
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        grid = self._generate_grid()
        
        if max_evaluations:
            grid = grid[:max_evaluations]
        
        history = []
        best_params = None
        best_score = float('inf')
        converged = False
        convergence_iteration = None
        
        def evaluate_params(params, iteration):
            """Evaluate single parameter set."""
            score = objective_function(params)
            return {
                'params': params.copy(),
                'score': score,
                'iteration': iteration
            }
        
        if parallel and len(grid) > 10:
            # Parallel evaluation
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(evaluate_params, params, i): i
                    for i, params in enumerate(grid)
                }
                
                for future in as_completed(futures):
                    result = future.result()
                    history.append(result)
                    
                    if result['score'] < best_score:
                        best_score = result['score']
                        best_params = result['params']
                    
                    # Check early stopping
                    if target_score and best_score <= target_score:
                        converged = True
                        convergence_iteration = len(history)
                        break
        else:
            # Sequential evaluation
            for i, params in enumerate(grid):
                result = evaluate_params(params, i)
                history.append(result)
                
                if result['score'] < best_score:
                    best_score = result['score']
                    best_params = result['params']
                
                # Check early stopping
                if target_score and best_score <= target_score:
                    converged = True
                    convergence_iteration = i
                    break
                
                # Check convergence
                if convergence_tolerance and self._check_convergence(
                    history, convergence_tolerance, convergence_window
                ):
                    converged = True
                    convergence_iteration = i
                    break
        
        # Sort history by iteration
        history.sort(key=lambda x: x['iteration'])
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            history=history,
            converged=converged,
            iterations=len(history),
            convergence_iteration=convergence_iteration,
            metrics={'time_elapsed': time.time() - start_time}
        )


class RandomSearchOptimizer(BaseOptimizer):
    """Random search parameter optimization."""
    
    def __init__(self, search_space: Dict[str, Dict[str, Any]],
                 constraints: Optional[List[Callable]] = None,
                 seed: Optional[int] = None):
        """
        Initialize random search optimizer.
        
        Args:
            search_space: Parameter definitions  
            constraints: List of constraint functions
            seed: Random seed for reproducibility
        """
        super().__init__(search_space, constraints)
        self.rng = np.random.RandomState(seed)
        if seed is not None:
            np.random.seed(seed)
    
    def set_seed(self, seed: int):
        """Set random seed."""
        self.rng = np.random.RandomState(seed)
        np.random.seed(seed)
    
    def optimize(self, objective_function: Callable,
                max_evaluations: int = 100,
                target_score: Optional[float] = None,
                patience: Optional[int] = None,
                min_improvement: float = 1e-6) -> OptimizationResult:
        """
        Run random search optimization.
        
        Args:
            objective_function: Function to minimize
            max_evaluations: Maximum number of evaluations
            target_score: Stop if score below target
            patience: Stop after no improvement for this many iterations
            min_improvement: Minimum improvement to reset patience
            
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        history = []
        best_params = None
        best_score = float('inf')
        best_iteration = 0
        converged = False
        
        for i in range(max_evaluations):
            # Sample valid parameters
            params = None
            for _ in range(1000):  # Max attempts
                candidate = self.sample_params()
                if self.validate_params(candidate):
                    params = candidate
                    break
            
            if params is None:
                logger.warning("Failed to sample valid parameters")
                continue
            
            # Evaluate
            score = objective_function(params)
            history.append({
                'params': params.copy(),
                'score': score,
                'iteration': i
            })
            
            # Update best
            if score < best_score - min_improvement:
                best_score = score
                best_params = params.copy()
                best_iteration = i
            
            # Check early stopping
            if target_score and best_score <= target_score:
                converged = True
                break
            
            # Check patience
            if patience and i - best_iteration >= patience:
                converged = True
                break
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            history=history,
            converged=converged,
            iterations=len(history),
            convergence_iteration=best_iteration if converged else None,
            metrics={
                'time_elapsed': time.time() - start_time,
                'evaluations_per_second': len(history) / (time.time() - start_time)
            }
        )


class SmartSearchOptimizer(BaseOptimizer):
    """Smart search with adaptive sampling strategies."""
    
    def __init__(self, search_space: Dict[str, Dict[str, Any]],
                 initial_samples: int = 10,
                 exploitation_rate: float = 0.7,
                 constraints: Optional[List[Callable]] = None,
                 track_metrics: bool = False):
        """
        Initialize smart search optimizer.
        
        Args:
            search_space: Parameter definitions
            initial_samples: Initial random samples
            exploitation_rate: Fraction of samples near best (0-1)
            constraints: List of constraint functions
            track_metrics: Track additional metrics
        """
        super().__init__(search_space, constraints)
        self.initial_samples = initial_samples
        self.exploitation_rate = exploitation_rate
        self.track_metrics = track_metrics
    
    def _sample_near_best(self, best_params: Dict[str, Any], 
                         radius: float = 0.1) -> Dict[str, Any]:
        """Sample parameters near best found so far."""
        new_params = {}
        
        for param, config in self.search_space.items():
            param_type = config['type']
            
            if param_type == 'float':
                # Sample within radius of best
                param_range = config['max'] - config['min']
                std = radius * param_range
                value = np.random.normal(best_params[param], std)
                value = np.clip(value, config['min'], config['max'])
                new_params[param] = value
                
            elif param_type == 'int':
                # Sample nearby integers
                param_range = config['max'] - config['min']
                std = max(1, int(radius * param_range))
                value = int(np.random.normal(best_params[param], std))
                value = np.clip(value, config['min'], config['max'])
                new_params[param] = value
                
            elif param_type == 'categorical':
                # Sometimes keep same, sometimes random
                if np.random.rand() > radius:
                    new_params[param] = best_params[param]
                else:
                    new_params[param] = np.random.choice(config['values'])
        
        return new_params
    
    def optimize(self, objective_function: Callable,
                max_evaluations: int = 100,
                target_score: Optional[float] = None,
                adaptive_radius: bool = True) -> OptimizationResult:
        """
        Run smart search optimization.
        
        Args:
            objective_function: Function to minimize
            max_evaluations: Maximum evaluations
            target_score: Stop if score below target
            adaptive_radius: Adapt search radius based on progress
            
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        history = []
        best_params = None
        best_score = float('inf')
        converged = False
        
        metrics = {
            'best_score_progression': [],
            'exploration_radius': []
        } if self.track_metrics else {}
        
        # Initial random exploration
        for i in range(min(self.initial_samples, max_evaluations)):
            params = None
            for _ in range(1000):
                candidate = self.sample_params()
                if self.validate_params(candidate):
                    params = candidate
                    break
            
            if params is None:
                continue
            
            score = objective_function(params)
            history.append({
                'params': params.copy(),
                'score': score,
                'iteration': i,
                'phase': 'exploration'
            })
            
            if score < best_score:
                best_score = score
                best_params = params.copy()
            
            if self.track_metrics:
                metrics['best_score_progression'].append(best_score)
        
        # Adaptive exploitation/exploration
        radius = 0.3  # Initial search radius
        no_improvement_count = 0
        
        for i in range(len(history), max_evaluations):
            # Decide exploration vs exploitation
            if np.random.rand() < self.exploitation_rate and best_params:
                # Exploit: sample near best
                params = None
                for _ in range(100):
                    candidate = self._sample_near_best(best_params, radius)
                    if self.validate_params(candidate):
                        params = candidate
                        break
                phase = 'exploitation'
            else:
                # Explore: random sample
                params = None
                for _ in range(100):
                    candidate = self.sample_params()
                    if self.validate_params(candidate):
                        params = candidate
                        break
                phase = 'exploration'
            
            if params is None:
                continue
            
            score = objective_function(params)
            history.append({
                'params': params.copy(),
                'score': score,
                'iteration': i,
                'phase': phase
            })
            
            # Update best
            if score < best_score:
                improvement = best_score - score
                best_score = score
                best_params = params.copy()
                no_improvement_count = 0
                
                # Reduce radius on improvement
                if adaptive_radius:
                    radius *= 0.95
            else:
                no_improvement_count += 1
                
                # Increase radius if stuck
                if adaptive_radius and no_improvement_count > 10:
                    radius = min(0.5, radius * 1.1)
                    no_improvement_count = 0
            
            if self.track_metrics:
                metrics['best_score_progression'].append(best_score)
                metrics['exploration_radius'].append(radius)
            
            # Check early stopping
            if target_score and best_score <= target_score:
                converged = True
                break
        
        # Calculate additional metrics
        if self.track_metrics:
            exploitation_count = sum(1 for h in history if h.get('phase') == 'exploitation')
            metrics['exploration_efficiency'] = exploitation_count / len(history)
            
            # Convergence rate
            if len(metrics['best_score_progression']) > 10:
                scores = np.array(metrics['best_score_progression'])
                improvements = -np.diff(scores)
                metrics['convergence_rate'] = np.mean(improvements[improvements > 0])
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            history=history,
            converged=converged,
            iterations=len(history),
            metrics=metrics
        )