"""
Bayesian optimization for trading system parameters.

Uses Gaussian Processes and acquisition functions to efficiently
explore the parameter space.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Callable, Optional, Union
import time
from dataclasses import dataclass
import logging
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
import warnings

from .parameter_optimizer import BaseOptimizer, OptimizationResult

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)


class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimization using Gaussian Processes."""
    
    def __init__(self, search_space: Dict[str, Dict[str, Any]],
                 initial_samples: int = 10,
                 acquisition_function: str = 'ei',
                 xi: float = 0.01,
                 kappa: float = 2.576,
                 kernel: Optional[Any] = None,
                 alpha: float = 1e-6,
                 normalize_y: bool = True,
                 constraints: Optional[List[Callable]] = None,
                 noise_level: Optional[float] = None):
        """
        Initialize Bayesian optimizer.
        
        Args:
            search_space: Parameter definitions
            initial_samples: Number of initial random samples
            acquisition_function: Acquisition function ('ei', 'ucb', 'poi')
            xi: Exploration parameter for EI/POI
            kappa: Exploration parameter for UCB
            kernel: GP kernel (default: Matern)
            alpha: Noise level for GP
            normalize_y: Normalize target values
            constraints: List of constraint functions
            noise_level: Expected noise in objective function
        """
        super().__init__(search_space, constraints)
        self.initial_samples = initial_samples
        self.acquisition_function = acquisition_function.lower()
        self.xi = xi
        self.kappa = kappa
        self.alpha = alpha if noise_level is None else noise_level ** 2
        self.normalize_y = normalize_y
        
        # Setup kernel
        if kernel is None:
            self.kernel = ConstantKernel(1.0) * Matern(
                length_scale=1.0, 
                length_scale_bounds=(1e-2, 1e2),
                nu=2.5
            )
        else:
            self.kernel = kernel
        
        # Parameter bounds for optimization
        self._setup_bounds()
        
        # Storage for observations
        self.X_observed = []
        self.y_observed = []
        
        # Gaussian Process
        self.gp = None
    
    def _setup_bounds(self):
        """Setup parameter bounds for acquisition optimization."""
        self.param_names = list(self.search_space.keys())
        self.bounds = []
        self.param_types = {}
        
        for param, config in self.search_space.items():
            self.param_types[param] = config['type']
            
            if config['type'] in ['float', 'int']:
                self.bounds.append((config['min'], config['max']))
            elif config['type'] == 'categorical':
                # Encode categorical as integer
                self.bounds.append((0, len(config['values']) - 1))
    
    def _params_to_array(self, params: Dict[str, Any]) -> np.ndarray:
        """Convert parameter dictionary to numpy array."""
        x = []
        
        for param in self.param_names:
            config = self.search_space[param]
            value = params[param]
            
            if config['type'] in ['float', 'int']:
                # Normalize to [0, 1]
                normalized = (value - config['min']) / (config['max'] - config['min'])
                x.append(normalized)
            elif config['type'] == 'categorical':
                # Encode as integer index
                idx = config['values'].index(value)
                normalized = idx / (len(config['values']) - 1) if len(config['values']) > 1 else 0
                x.append(normalized)
        
        return np.array(x)
    
    def _array_to_params(self, x: np.ndarray) -> Dict[str, Any]:
        """Convert numpy array to parameter dictionary."""
        params = {}
        
        for i, param in enumerate(self.param_names):
            config = self.search_space[param]
            normalized = x[i]
            
            if config['type'] == 'float':
                # Denormalize
                value = normalized * (config['max'] - config['min']) + config['min']
                params[param] = float(value)
            elif config['type'] == 'int':
                # Denormalize and round
                value = normalized * (config['max'] - config['min']) + config['min']
                params[param] = int(round(value))
            elif config['type'] == 'categorical':
                # Decode from index
                idx = int(round(normalized * (len(config['values']) - 1)))
                idx = np.clip(idx, 0, len(config['values']) - 1)
                params[param] = config['values'][idx]
        
        return params
    
    def _expected_improvement(self, X: np.ndarray, X_sample: np.ndarray, 
                            y_sample: np.ndarray, gp: Any, 
                            xi: float = 0.01) -> np.ndarray:
        """Calculate expected improvement acquisition function."""
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        
        # Best observed value
        if self.normalize_y:
            y_max = np.max(y_sample)
        else:
            y_max = np.max(y_sample)
        
        # Handle minimization
        y_max = -y_max
        mu = -mu
        
        # Calculate EI
        with np.errstate(divide='warn'):
            imp = mu - y_max - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei
    
    def _upper_confidence_bound(self, X: np.ndarray, gp: Any, 
                               kappa: float = 2.576) -> np.ndarray:
        """Calculate upper confidence bound acquisition function."""
        mu, sigma = gp.predict(X, return_std=True)
        
        # Handle minimization by negating
        return -(mu - kappa * sigma)
    
    def _probability_of_improvement(self, X: np.ndarray, X_sample: np.ndarray,
                                   y_sample: np.ndarray, gp: Any,
                                   xi: float = 0.01) -> np.ndarray:
        """Calculate probability of improvement acquisition function."""
        mu, sigma = gp.predict(X, return_std=True)
        
        # Best observed value
        y_max = np.max(y_sample)
        
        # Handle minimization
        y_max = -y_max
        mu = -mu
        
        # Calculate POI
        with np.errstate(divide='warn'):
            Z = (mu - y_max - xi) / sigma
            poi = norm.cdf(Z)
            poi[sigma == 0.0] = 0.0
        
        return poi
    
    def _optimize_acquisition(self, n_points: int = 1) -> List[np.ndarray]:
        """Find points that maximize the acquisition function."""
        dim = len(self.param_names)
        
        # Acquisition function to minimize (negative for maximization)
        def acquisition(x):
            x_array = x.reshape(1, -1)
            
            if self.acquisition_function == 'ei':
                return -self._expected_improvement(
                    x_array, self.X_observed, self.y_observed, 
                    self.gp, self.xi
                ).ravel()[0]
            elif self.acquisition_function == 'ucb':
                return -self._upper_confidence_bound(
                    x_array, self.gp, self.kappa
                ).ravel()[0]
            elif self.acquisition_function == 'poi':
                return -self._probability_of_improvement(
                    x_array, self.X_observed, self.y_observed,
                    self.gp, self.xi
                ).ravel()[0]
            else:
                raise ValueError(f"Unknown acquisition function: {self.acquisition_function}")
        
        # Multiple random restarts
        n_restarts = 25
        best_points = []
        
        for _ in range(n_points):
            best_x = None
            best_acq = float('inf')
            
            for _ in range(n_restarts):
                # Random starting point
                x0 = np.random.uniform(0, 1, dim)
                
                # Optimize
                try:
                    res = minimize(
                        acquisition,
                        x0,
                        bounds=[(0, 1)] * dim,
                        method='L-BFGS-B'
                    )
                    
                    if res.fun < best_acq:
                        best_acq = res.fun
                        best_x = res.x
                except Exception as e:
                    logger.debug(f"Acquisition optimization failed: {e}")
                    continue
            
            if best_x is not None:
                best_points.append(best_x)
            else:
                # Fallback to random
                best_points.append(np.random.uniform(0, 1, dim))
        
        return best_points
    
    def optimize(self, objective_function: Callable,
                max_evaluations: int = 100,
                target_score: Optional[float] = None,
                convergence_tolerance: Optional[float] = None,
                convergence_window: int = 10,
                initial_tolerance: Optional[float] = None,
                final_tolerance: Optional[float] = None,
                batch_size: int = 1,
                time_limit: Optional[float] = None,
                relative_tolerance: Optional[float] = None) -> OptimizationResult:
        """
        Run Bayesian optimization.
        
        Args:
            objective_function: Function to minimize
            max_evaluations: Maximum function evaluations
            target_score: Stop if score below target
            convergence_tolerance: Convergence tolerance
            convergence_window: Window for convergence check
            initial_tolerance: Initial tolerance for adaptive convergence
            final_tolerance: Final tolerance for adaptive convergence
            batch_size: Number of points to evaluate in parallel
            time_limit: Time limit in seconds
            relative_tolerance: Relative convergence tolerance
            
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        history = []
        best_params = None
        best_score = float('inf')
        converged = False
        convergence_iteration = None
        
        # Initial random sampling
        logger.info(f"Starting with {self.initial_samples} initial samples")
        
        for i in range(min(self.initial_samples, max_evaluations)):
            # Sample valid parameters
            for _ in range(1000):
                params = self.sample_params()
                if self.validate_params(params):
                    break
            else:
                logger.warning("Failed to sample valid parameters")
                continue
            
            # Evaluate
            try:
                score = objective_function(params)
            except Exception as e:
                logger.error(f"Error evaluating parameters: {e}")
                continue
            
            # Store observation
            x = self._params_to_array(params)
            self.X_observed.append(x)
            self.y_observed.append(score)
            
            # Update best
            if score < best_score:
                best_score = score
                best_params = params.copy()
            
            # Record history
            history.append({
                'params': params.copy(),
                'score': score,
                'iteration': i,
                'phase': 'initial'
            })
            
            # Check early stopping
            if target_score and best_score <= target_score:
                converged = True
                convergence_iteration = i
                break
        
        # Convert observations to arrays
        self.X_observed = np.array(self.X_observed)
        self.y_observed = np.array(self.y_observed)
        
        # Bayesian optimization loop
        for i in range(len(history), max_evaluations, batch_size):
            # Check time limit
            if time_limit and (time.time() - start_time) > time_limit:
                logger.info("Time limit reached")
                break
            
            # Fit Gaussian Process
            self.gp = GaussianProcessRegressor(
                kernel=self.kernel,
                alpha=self.alpha,
                normalize_y=self.normalize_y,
                n_restarts_optimizer=5
            )
            
            try:
                self.gp.fit(self.X_observed, self.y_observed)
            except Exception as e:
                logger.error(f"GP fitting failed: {e}")
                # Fallback to random sampling
                for j in range(min(batch_size, max_evaluations - i)):
                    params = self.sample_params()
                    if self.validate_params(params):
                        score = objective_function(params)
                        x = self._params_to_array(params)
                        
                        self.X_observed = np.vstack([self.X_observed, x])
                        self.y_observed = np.append(self.y_observed, score)
                        
                        if score < best_score:
                            best_score = score
                            best_params = params.copy()
                        
                        history.append({
                            'params': params.copy(),
                            'score': score,
                            'iteration': i + j,
                            'phase': 'fallback'
                        })
                continue
            
            # Find next points to evaluate
            next_points = self._optimize_acquisition(n_points=batch_size)
            
            # Evaluate points
            for j, x in enumerate(next_points):
                if i + j >= max_evaluations:
                    break
                
                # Convert to parameters
                params = self._array_to_params(x)
                
                # Validate
                if not self.validate_params(params):
                    # Try to find nearby valid point
                    for _ in range(10):
                        x_perturbed = x + np.random.normal(0, 0.01, len(x))
                        x_perturbed = np.clip(x_perturbed, 0, 1)
                        params = self._array_to_params(x_perturbed)
                        if self.validate_params(params):
                            x = x_perturbed
                            break
                    else:
                        logger.warning("Failed to find valid parameters")
                        continue
                
                # Evaluate
                try:
                    score = objective_function(params)
                except Exception as e:
                    logger.error(f"Error evaluating parameters: {e}")
                    continue
                
                # Update observations
                self.X_observed = np.vstack([self.X_observed, x])
                self.y_observed = np.append(self.y_observed, score)
                
                # Update best
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
                
                # Record history
                history.append({
                    'params': params.copy(),
                    'score': score,
                    'iteration': i + j,
                    'phase': 'bayesian',
                    'acquisition': self.acquisition_function
                })
                
                # Check early stopping
                if target_score and best_score <= target_score:
                    converged = True
                    convergence_iteration = i + j
                    break
            
            # Check convergence
            if convergence_tolerance and len(history) >= convergence_window:
                recent_scores = [h['score'] for h in history[-convergence_window:]]
                
                if max(recent_scores) - min(recent_scores) < convergence_tolerance:
                    converged = True
                    convergence_iteration = len(history) - 1
                    break
            
            # Check relative tolerance
            if relative_tolerance and len(history) >= convergence_window:
                recent_scores = [h['score'] for h in history[-convergence_window:]]
                if abs(recent_scores[0]) > 1e-10:
                    rel_change = abs(recent_scores[-1] - recent_scores[0]) / abs(recent_scores[0])
                    if rel_change < relative_tolerance:
                        converged = True
                        convergence_iteration = len(history) - 1
                        break
            
            if converged:
                break
        
        # Calculate final metrics
        elapsed_time = time.time() - start_time
        
        metrics = {
            'time_elapsed': elapsed_time,
            'gp_kernel': str(self.kernel),
            'acquisition_function': self.acquisition_function,
            'initial_samples': self.initial_samples
        }
        
        # Add adaptive tolerance history if used
        if initial_tolerance and final_tolerance:
            n_iterations = len(history)
            tolerance_history = []
            for i in range(n_iterations):
                progress = i / max(1, n_iterations - 1)
                tol = initial_tolerance * (1 - progress) + final_tolerance * progress
                tolerance_history.append(tol)
            metrics['tolerance_history'] = tolerance_history
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            history=history,
            converged=converged,
            iterations=len(history),
            convergence_iteration=convergence_iteration,
            metrics=metrics,
            early_stopped=target_score and best_score <= target_score,
            time_stopped=time_limit and elapsed_time >= time_limit
        )