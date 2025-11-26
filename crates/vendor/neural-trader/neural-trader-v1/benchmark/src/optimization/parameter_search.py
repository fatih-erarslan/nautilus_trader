"""Hyperparameter tuning framework for trading strategies.

This module provides comprehensive parameter search capabilities including
grid search, random search, and advanced sampling techniques.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from enum import Enum
import itertools
from scipy.stats import uniform, loguniform, randint
from sklearn.model_selection import ParameterGrid, ParameterSampler
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

logger = logging.getLogger(__name__)


class ParameterType(Enum):
    """Types of parameters in search space."""
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


@dataclass
class Parameter:
    """Parameter definition for search space."""
    name: str
    type: ParameterType
    low: Optional[Union[float, int]] = None
    high: Optional[Union[float, int]] = None
    choices: Optional[List[Any]] = None
    distribution: Optional[str] = 'uniform'  # 'uniform', 'log-uniform', 'normal'
    

@dataclass
class SearchResult:
    """Results from parameter search."""
    best_params: Dict[str, Any]
    best_score: float
    all_results: pd.DataFrame
    search_space_coverage: float
    n_evaluations: int
    search_time: float
    convergence_curve: List[float]


class ParameterSpace:
    """Defines and manages parameter search space."""
    
    def __init__(self, parameters: List[Parameter]):
        """Initialize parameter space.
        
        Args:
            parameters: List of parameter definitions
        """
        self.parameters = {p.name: p for p in parameters}
        self._validate_parameters()
        
    def _validate_parameters(self):
        """Validate parameter definitions."""
        for name, param in self.parameters.items():
            if param.type in [ParameterType.CONTINUOUS, ParameterType.INTEGER]:
                if param.low is None or param.high is None:
                    raise ValueError(f"Parameter {name} must have low and high bounds")
                if param.low >= param.high:
                    raise ValueError(f"Parameter {name} low must be less than high")
            elif param.type == ParameterType.CATEGORICAL:
                if not param.choices:
                    raise ValueError(f"Categorical parameter {name} must have choices")
            elif param.type == ParameterType.BOOLEAN:
                param.choices = [True, False]
                
    def sample(self, n_samples: int = 1, method: str = 'random') -> List[Dict[str, Any]]:
        """Sample parameters from the space.
        
        Args:
            n_samples: Number of samples to generate
            method: Sampling method ('random', 'latin_hypercube', 'sobol')
            
        Returns:
            List of parameter dictionaries
        """
        if method == 'random':
            return self._random_sample(n_samples)
        elif method == 'latin_hypercube':
            return self._latin_hypercube_sample(n_samples)
        elif method == 'sobol':
            return self._sobol_sample(n_samples)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
            
    def _random_sample(self, n_samples: int) -> List[Dict[str, Any]]:
        """Random sampling from parameter space."""
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            for name, param in self.parameters.items():
                if param.type == ParameterType.CONTINUOUS:
                    if param.distribution == 'uniform':
                        value = uniform.rvs(param.low, param.high - param.low)
                    elif param.distribution == 'log-uniform':
                        value = loguniform.rvs(param.low, param.high)
                    else:
                        value = uniform.rvs(param.low, param.high - param.low)
                elif param.type == ParameterType.INTEGER:
                    value = randint.rvs(param.low, param.high + 1)
                elif param.type in [ParameterType.CATEGORICAL, ParameterType.BOOLEAN]:
                    value = np.random.choice(param.choices)
                    
                sample[name] = value
            samples.append(sample)
            
        return samples
        
    def _latin_hypercube_sample(self, n_samples: int) -> List[Dict[str, Any]]:
        """Latin hypercube sampling for better coverage."""
        from scipy.stats import qmc
        
        # Get continuous parameters
        continuous_params = [
            (name, param) for name, param in self.parameters.items()
            if param.type == ParameterType.CONTINUOUS
        ]
        
        if not continuous_params:
            return self._random_sample(n_samples)
            
        # Generate LHS samples for continuous parameters
        sampler = qmc.LatinHypercube(d=len(continuous_params))
        lhs_samples = sampler.random(n=n_samples)
        
        samples = []
        for i in range(n_samples):
            sample = {}
            
            # Add continuous parameters from LHS
            for j, (name, param) in enumerate(continuous_params):
                # Scale from [0, 1] to parameter range
                if param.distribution == 'uniform':
                    value = param.low + lhs_samples[i, j] * (param.high - param.low)
                elif param.distribution == 'log-uniform':
                    log_low = np.log(param.low)
                    log_high = np.log(param.high)
                    value = np.exp(log_low + lhs_samples[i, j] * (log_high - log_low))
                else:
                    value = param.low + lhs_samples[i, j] * (param.high - param.low)
                    
                sample[name] = value
                
            # Add discrete parameters randomly
            for name, param in self.parameters.items():
                if param.type == ParameterType.INTEGER:
                    sample[name] = randint.rvs(param.low, param.high + 1)
                elif param.type in [ParameterType.CATEGORICAL, ParameterType.BOOLEAN]:
                    sample[name] = np.random.choice(param.choices)
                    
            samples.append(sample)
            
        return samples
        
    def _sobol_sample(self, n_samples: int) -> List[Dict[str, Any]]:
        """Sobol sequence sampling for low-discrepancy coverage."""
        from scipy.stats import qmc
        
        # Get all numeric parameters
        numeric_params = [
            (name, param) for name, param in self.parameters.items()
            if param.type in [ParameterType.CONTINUOUS, ParameterType.INTEGER]
        ]
        
        if not numeric_params:
            return self._random_sample(n_samples)
            
        # Generate Sobol samples
        sampler = qmc.Sobol(d=len(numeric_params), scramble=True)
        sobol_samples = sampler.random(n=n_samples)
        
        samples = []
        for i in range(n_samples):
            sample = {}
            
            # Add numeric parameters from Sobol
            for j, (name, param) in enumerate(numeric_params):
                if param.type == ParameterType.CONTINUOUS:
                    if param.distribution == 'uniform':
                        value = param.low + sobol_samples[i, j] * (param.high - param.low)
                    elif param.distribution == 'log-uniform':
                        log_low = np.log(param.low)
                        log_high = np.log(param.high)
                        value = np.exp(log_low + sobol_samples[i, j] * (log_high - log_low))
                    else:
                        value = param.low + sobol_samples[i, j] * (param.high - param.low)
                else:  # INTEGER
                    value = int(param.low + sobol_samples[i, j] * (param.high - param.low + 1))
                    value = min(value, param.high)  # Ensure within bounds
                    
                sample[name] = value
                
            # Add categorical parameters randomly
            for name, param in self.parameters.items():
                if param.type in [ParameterType.CATEGORICAL, ParameterType.BOOLEAN]:
                    sample[name] = np.random.choice(param.choices)
                    
            samples.append(sample)
            
        return samples
        
    def get_grid(self, resolution: Dict[str, int] = None) -> List[Dict[str, Any]]:
        """Generate grid of parameters.
        
        Args:
            resolution: Number of points for each continuous parameter
            
        Returns:
            List of all parameter combinations
        """
        if resolution is None:
            resolution = {}
            
        param_values = {}
        
        for name, param in self.parameters.items():
            if param.type == ParameterType.CONTINUOUS:
                n_points = resolution.get(name, 10)
                if param.distribution == 'log-uniform':
                    values = np.logspace(
                        np.log10(param.low),
                        np.log10(param.high),
                        n_points
                    )
                else:
                    values = np.linspace(param.low, param.high, n_points)
                param_values[name] = values
            elif param.type == ParameterType.INTEGER:
                n_points = min(resolution.get(name, 10), param.high - param.low + 1)
                values = np.linspace(param.low, param.high, n_points, dtype=int)
                param_values[name] = np.unique(values)
            elif param.type in [ParameterType.CATEGORICAL, ParameterType.BOOLEAN]:
                param_values[name] = param.choices
                
        # Generate all combinations
        grid = list(ParameterGrid(param_values))
        
        return grid


class HyperparameterSearch:
    """Main hyperparameter search engine."""
    
    def __init__(
        self,
        objective_function: Callable,
        parameter_space: ParameterSpace,
        n_jobs: int = -1
    ):
        """Initialize hyperparameter search.
        
        Args:
            objective_function: Function to optimize
            parameter_space: Parameter search space
            n_jobs: Number of parallel jobs
        """
        self.objective_function = objective_function
        self.parameter_space = parameter_space
        self.n_jobs = n_jobs if n_jobs > 0 else asyncio.cpu_count()
        
        self.results: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = float('-inf')
        
    async def search(
        self,
        method: str = 'random',
        n_iter: int = 100,
        callbacks: Optional[List[Callable]] = None
    ) -> SearchResult:
        """Run hyperparameter search.
        
        Args:
            method: Search method ('grid', 'random', 'halving', 'hyperband')
            n_iter: Number of iterations (for non-grid methods)
            callbacks: List of callback functions
            
        Returns:
            Search results
        """
        start_time = datetime.now()
        
        if method == 'grid':
            await self._grid_search(callbacks)
        elif method == 'random':
            await self._random_search(n_iter, callbacks)
        elif method == 'halving':
            await self._halving_search(n_iter, callbacks)
        elif method == 'hyperband':
            await self._hyperband_search(n_iter, callbacks)
        else:
            raise ValueError(f"Unknown search method: {method}")
            
        search_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare results
        results_df = pd.DataFrame(self.results)
        
        # Calculate search space coverage
        coverage = self._calculate_coverage()
        
        # Get convergence curve
        convergence_curve = self._get_convergence_curve()
        
        return SearchResult(
            best_params=self.best_params,
            best_score=self.best_score,
            all_results=results_df,
            search_space_coverage=coverage,
            n_evaluations=len(self.results),
            search_time=search_time,
            convergence_curve=convergence_curve
        )
        
    async def _grid_search(self, callbacks: Optional[List[Callable]]):
        """Exhaustive grid search."""
        grid = self.parameter_space.get_grid()
        logger.info(f"Grid search with {len(grid)} parameter combinations")
        
        # Evaluate all combinations in parallel
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            
            for params in grid:
                future = executor.submit(self._evaluate_params, params)
                futures.append((params, future))
                
            # Process results
            for params, future in futures:
                score = future.result()
                await self._process_result(params, score, callbacks)
                
    async def _random_search(self, n_iter: int, callbacks: Optional[List[Callable]]):
        """Random parameter search."""
        logger.info(f"Random search with {n_iter} iterations")
        
        # Sample parameters
        param_samples = self.parameter_space.sample(n_iter, method='random')
        
        # Evaluate in batches
        batch_size = self.n_jobs
        for i in range(0, n_iter, batch_size):
            batch = param_samples[i:i + batch_size]
            await self._evaluate_batch(batch, callbacks)
            
    async def _halving_search(self, n_iter: int, callbacks: Optional[List[Callable]]):
        """Successive halving for efficient search."""
        n_configs = n_iter
        min_resource = 1
        max_resource = 27  # e.g., epochs or data size
        eta = 3  # Halving rate
        
        # Calculate number of rounds
        s_max = int(np.log(max_resource / min_resource) / np.log(eta))
        
        for s in reversed(range(s_max + 1)):
            # Initial number of configurations
            n = int(np.ceil(n_configs * (eta ** s) / (s_max + 1)))
            
            # Initial resource allocation
            r = max_resource / (eta ** s)
            
            # Sample n configurations
            configs = self.parameter_space.sample(n, method='random')
            
            # Successive halving
            for i in range(s + 1):
                # Number of configs to keep
                n_i = int(n / (eta ** i))
                r_i = int(r * (eta ** i))
                
                # Evaluate configurations with r_i resources
                scores = []
                for config in configs[:n_i]:
                    score = await self._evaluate_with_budget(config, r_i)
                    scores.append(score)
                    await self._process_result(config, score, callbacks)
                    
                # Keep top 1/eta configurations
                if i < s:
                    indices = np.argsort(scores)[::-1][:int(n_i / eta)]
                    configs = [configs[idx] for idx in indices]
                    
    async def _hyperband_search(self, n_iter: int, callbacks: Optional[List[Callable]]):
        """Hyperband: adaptive resource allocation."""
        max_resource = 81  # Maximum budget per config
        eta = 3  # Halving rate
        
        # Number of unique executions of successive halving
        s_max = int(np.log(max_resource) / np.log(eta))
        B = (s_max + 1) * max_resource  # Total budget
        
        for s in reversed(range(s_max + 1)):
            # Number of configs and resources for this round
            n = int(np.ceil(B / max_resource / (s + 1) * (eta ** s)))
            r = max_resource * (eta ** (-s))
            
            # Run successive halving
            configs = self.parameter_space.sample(n, method='sobol')
            
            for i in range(s + 1):
                # Number of configs to keep
                n_i = int(n * (eta ** (-i)))
                r_i = int(r * (eta ** i))
                
                # Evaluate configurations
                scores = []
                for config in configs[:n_i]:
                    score = await self._evaluate_with_budget(config, r_i)
                    scores.append(score)
                    await self._process_result(config, score, callbacks)
                    
                # Select top configurations
                if i < s:
                    indices = np.argsort(scores)[::-1][:int(n_i / eta)]
                    configs = [configs[idx] for idx in indices]
                    
    def _evaluate_params(self, params: Dict[str, Any]) -> float:
        """Evaluate parameters."""
        try:
            return float(self.objective_function(params))
        except Exception as e:
            logger.error(f"Error evaluating params {params}: {str(e)}")
            return float('-inf')
            
    async def _evaluate_batch(
        self,
        batch: List[Dict[str, Any]],
        callbacks: Optional[List[Callable]]
    ):
        """Evaluate batch of parameters in parallel."""
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            
            for params in batch:
                future = executor.submit(self._evaluate_params, params)
                futures.append((params, future))
                
            # Process results
            for params, future in futures:
                score = future.result()
                await self._process_result(params, score, callbacks)
                
    async def _evaluate_with_budget(
        self,
        params: Dict[str, Any],
        budget: int
    ) -> float:
        """Evaluate parameters with limited budget."""
        # Modify objective function to use limited resources
        # This is problem-specific (e.g., fewer epochs, less data)
        return self._evaluate_params(params)
        
    async def _process_result(
        self,
        params: Dict[str, Any],
        score: float,
        callbacks: Optional[List[Callable]]
    ):
        """Process evaluation result."""
        result = {
            'params': params,
            'score': score,
            'iteration': len(self.results),
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(result)
        
        # Update best
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
            logger.info(f"New best score: {score:.6f}")
            
        # Execute callbacks
        if callbacks:
            for callback in callbacks:
                callback(result)
                
    def _calculate_coverage(self) -> float:
        """Calculate search space coverage."""
        if not self.results:
            return 0.0
            
        # For each parameter, calculate unique values explored
        coverage_scores = []
        
        for name, param in self.parameter_space.parameters.items():
            values = [r['params'].get(name) for r in self.results]
            values = [v for v in values if v is not None]
            
            if not values:
                continue
                
            if param.type == ParameterType.CONTINUOUS:
                # Calculate coverage as range explored / total range
                if len(values) > 1:
                    explored_range = max(values) - min(values)
                    total_range = param.high - param.low
                    coverage = explored_range / total_range
                else:
                    coverage = 0.0
            elif param.type == ParameterType.INTEGER:
                # Calculate coverage as unique values / possible values
                unique_values = len(set(values))
                possible_values = param.high - param.low + 1
                coverage = unique_values / possible_values
            elif param.type in [ParameterType.CATEGORICAL, ParameterType.BOOLEAN]:
                # Calculate coverage as unique values / total choices
                unique_values = len(set(values))
                total_choices = len(param.choices)
                coverage = unique_values / total_choices
                
            coverage_scores.append(coverage)
            
        return np.mean(coverage_scores) if coverage_scores else 0.0
        
    def _get_convergence_curve(self) -> List[float]:
        """Get best score over iterations."""
        if not self.results:
            return []
            
        sorted_results = sorted(self.results, key=lambda x: x['iteration'])
        best_scores = []
        current_best = float('-inf')
        
        for result in sorted_results:
            current_best = max(current_best, result['score'])
            best_scores.append(current_best)
            
        return best_scores


class AdaptiveParameterTuner:
    """Adaptive parameter tuning based on online performance."""
    
    def __init__(
        self,
        base_params: Dict[str, Any],
        adaptation_rate: float = 0.1
    ):
        """Initialize adaptive tuner.
        
        Args:
            base_params: Base parameter values
            adaptation_rate: Learning rate for adaptations
        """
        self.base_params = base_params
        self.current_params = base_params.copy()
        self.adaptation_rate = adaptation_rate
        
        self.performance_history: List[float] = []
        self.parameter_history: List[Dict[str, Any]] = []
        
    def adapt(
        self,
        performance: float,
        gradient: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Adapt parameters based on performance.
        
        Args:
            performance: Current performance metric
            gradient: Optional gradient information
            
        Returns:
            Updated parameters
        """
        self.performance_history.append(performance)
        self.parameter_history.append(self.current_params.copy())
        
        if gradient:
            # Gradient-based update
            for param, grad in gradient.items():
                if param in self.current_params:
                    self.current_params[param] += self.adaptation_rate * grad
        else:
            # Performance-based adaptation
            if len(self.performance_history) > 1:
                perf_change = performance - self.performance_history[-2]
                
                if perf_change > 0:
                    # Keep current direction
                    self.adaptation_rate *= 1.1
                else:
                    # Reverse direction with smaller step
                    self.adaptation_rate *= 0.5
                    
        return self.current_params
        
    def reset(self):
        """Reset to base parameters."""
        self.current_params = self.base_params.copy()
        self.performance_history.clear()
        self.parameter_history.clear()


# Convenience functions
def create_parameter_space(param_dict: Dict[str, Any]) -> ParameterSpace:
    """Create parameter space from dictionary specification.
    
    Args:
        param_dict: Dictionary mapping parameter names to specifications
        
    Returns:
        ParameterSpace object
    """
    parameters = []
    
    for name, spec in param_dict.items():
        if isinstance(spec, tuple) and len(spec) == 2:
            # Continuous parameter (low, high)
            param = Parameter(
                name=name,
                type=ParameterType.CONTINUOUS,
                low=spec[0],
                high=spec[1]
            )
        elif isinstance(spec, list):
            # Categorical parameter
            param = Parameter(
                name=name,
                type=ParameterType.CATEGORICAL,
                choices=spec
            )
        elif isinstance(spec, dict):
            # Full specification
            param_type = ParameterType(spec.get('type', 'continuous'))
            param = Parameter(
                name=name,
                type=param_type,
                low=spec.get('low'),
                high=spec.get('high'),
                choices=spec.get('choices'),
                distribution=spec.get('distribution', 'uniform')
            )
        else:
            raise ValueError(f"Invalid parameter specification for {name}")
            
        parameters.append(param)
        
    return ParameterSpace(parameters)