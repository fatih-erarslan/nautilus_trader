"""Bayesian optimization implementation for hyperparameter tuning.

This module provides a comprehensive Bayesian optimization framework
using Gaussian processes for efficient parameter exploration.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Tuple
import logging
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
import warnings
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class AcquisitionFunction(Enum):
    """Acquisition function types."""
    EXPECTED_IMPROVEMENT = "expected_improvement"
    PROBABILITY_IMPROVEMENT = "probability_improvement"
    UPPER_CONFIDENCE_BOUND = "upper_confidence_bound"
    ENTROPY_SEARCH = "entropy_search"


@dataclass
class BayesianOptimizationResult:
    """Result from Bayesian optimization."""
    best_params: Dict[str, Any]
    best_score: float
    all_evaluations: List[Dict[str, Any]]
    gp_model: GaussianProcessRegressor
    acquisition_values: List[float]
    convergence_history: List[float]
    n_iterations: int


class BayesianOptimization:
    """Bayesian optimization using Gaussian processes."""
    
    def __init__(
        self,
        objective_function: Callable,
        parameter_space: Dict[str, Any],
        acquisition_function: AcquisitionFunction = AcquisitionFunction.EXPECTED_IMPROVEMENT,
        kernel: Optional[Any] = None,
        n_initial_points: int = 10,
        alpha: float = 1e-6,
        normalize_y: bool = True,
        random_state: Optional[int] = None
    ):
        """Initialize Bayesian optimization.
        
        Args:
            objective_function: Function to optimize
            parameter_space: Dictionary defining parameter bounds
            acquisition_function: Acquisition function to use
            kernel: GP kernel (default: Matern)
            n_initial_points: Number of initial random points
            alpha: GP noise level
            normalize_y: Whether to normalize targets
            random_state: Random seed
        """
        self.objective_function = objective_function
        self.parameter_space = parameter_space
        self.acquisition_function = acquisition_function
        self.n_initial_points = n_initial_points
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.random_state = random_state
        
        # Set random seed
        np.random.seed(random_state)
        
        # Initialize parameter bounds
        self.param_names = list(parameter_space.keys())
        self.bounds = self._process_parameter_space()
        self.n_dims = len(self.bounds)
        
        # Initialize Gaussian Process
        if kernel is None:
            kernel = ConstantKernel(1.0) * Matern(
                length_scale=1.0,
                length_scale_bounds=(1e-2, 1e2),
                nu=2.5
            )
            
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=normalize_y,
            n_restarts_optimizer=5,
            random_state=random_state
        )
        
        # Storage for evaluations
        self.X_observed = []
        self.y_observed = []
        self.evaluations = []
        self.best_params = None
        self.best_score = float('-inf')
        
        logger.info(f"Initialized Bayesian optimization with {self.n_dims} parameters")
        
    def _process_parameter_space(self) -> List[Tuple[float, float]]:
        """Process parameter space into bounds."""
        bounds = []
        
        for param_name, param_def in self.parameter_space.items():
            if isinstance(param_def, tuple) and len(param_def) == 2:
                # Continuous parameter
                bounds.append((float(param_def[0]), float(param_def[1])))
            elif isinstance(param_def, list):
                # Categorical parameter - map to indices
                bounds.append((0, len(param_def) - 1))
            else:
                raise ValueError(f"Invalid parameter definition for {param_name}")
                
        return bounds
        
    def _encode_params(self, params: Dict[str, Any]) -> np.ndarray:
        """Encode parameters to numerical array."""
        x = np.zeros(self.n_dims)
        
        for i, param_name in enumerate(self.param_names):
            param_def = self.parameter_space[param_name]
            param_value = params[param_name]
            
            if isinstance(param_def, tuple):
                # Continuous parameter - already numerical
                x[i] = param_value
            elif isinstance(param_def, list):
                # Categorical parameter - encode as index
                x[i] = param_def.index(param_value)
                
        return x
        
    def _decode_params(self, x: np.ndarray) -> Dict[str, Any]:
        """Decode numerical array to parameters."""
        params = {}
        
        for i, param_name in enumerate(self.param_names):
            param_def = self.parameter_space[param_name]
            
            if isinstance(param_def, tuple):
                # Continuous parameter
                params[param_name] = float(x[i])
            elif isinstance(param_def, list):
                # Categorical parameter
                idx = int(round(x[i]))
                idx = max(0, min(idx, len(param_def) - 1))
                params[param_name] = param_def[idx]
                
        return params
        
    def _sample_initial_points(self) -> List[np.ndarray]:
        """Sample initial points using Latin Hypercube Sampling."""
        from scipy.stats import qmc
        
        # Use Latin Hypercube Sampling for better coverage
        sampler = qmc.LatinHypercube(d=self.n_dims, seed=self.random_state)
        samples = sampler.random(n=self.n_initial_points)
        
        # Scale to parameter bounds
        scaled_samples = []
        for sample in samples:
            scaled = np.zeros(self.n_dims)
            for i, (low, high) in enumerate(self.bounds):
                scaled[i] = low + sample[i] * (high - low)
            scaled_samples.append(scaled)
            
        return scaled_samples
        
    def _evaluate_point(self, x: np.ndarray) -> float:
        """Evaluate objective function at point."""
        params = self._decode_params(x)
        
        try:
            score = self.objective_function(params)
            return float(score)
        except Exception as e:
            logger.error(f"Error evaluating {params}: {str(e)}")
            return float('-inf')
            
    def _acquisition_expected_improvement(
        self,
        x: np.ndarray,
        xi: float = 0.01
    ) -> float:
        """Expected improvement acquisition function."""
        if len(self.y_observed) == 0:
            return 0.0
            
        # Predict mean and variance
        mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
        mu = mu[0]
        sigma = sigma[0]
        
        if sigma == 0:
            return 0.0
            
        # Best observed value
        f_best = max(self.y_observed)
        
        # Expected improvement
        z = (mu - f_best - xi) / sigma
        ei = (mu - f_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        
        return ei
        
    def _acquisition_probability_improvement(
        self,
        x: np.ndarray,
        xi: float = 0.01
    ) -> float:
        """Probability of improvement acquisition function."""
        if len(self.y_observed) == 0:
            return 0.0
            
        mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
        mu = mu[0]
        sigma = sigma[0]
        
        if sigma == 0:
            return 0.0
            
        f_best = max(self.y_observed)
        z = (mu - f_best - xi) / sigma
        
        return norm.cdf(z)
        
    def _acquisition_upper_confidence_bound(
        self,
        x: np.ndarray,
        kappa: float = 2.576  # 99% confidence
    ) -> float:
        """Upper confidence bound acquisition function."""
        if len(self.y_observed) == 0:
            return 0.0
            
        mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
        mu = mu[0]
        sigma = sigma[0]
        
        return mu + kappa * sigma
        
    def _acquisition_entropy_search(self, x: np.ndarray) -> float:
        """Entropy search acquisition function (simplified)."""
        if len(self.y_observed) == 0:
            return 0.0
            
        mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
        sigma = sigma[0]
        
        # Approximate entropy reduction
        if sigma > 0:
            return 0.5 * np.log(2 * np.pi * np.e * sigma**2)
        else:
            return 0.0
            
    def _compute_acquisition(self, x: np.ndarray) -> float:
        """Compute acquisition function value."""
        if self.acquisition_function == AcquisitionFunction.EXPECTED_IMPROVEMENT:
            return self._acquisition_expected_improvement(x)
        elif self.acquisition_function == AcquisitionFunction.PROBABILITY_IMPROVEMENT:
            return self._acquisition_probability_improvement(x)
        elif self.acquisition_function == AcquisitionFunction.UPPER_CONFIDENCE_BOUND:
            return self._acquisition_upper_confidence_bound(x)
        elif self.acquisition_function == AcquisitionFunction.ENTROPY_SEARCH:
            return self._acquisition_entropy_search(x)
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_function}")
            
    def _optimize_acquisition(self, n_restarts: int = 10) -> np.ndarray:
        """Optimize acquisition function to find next point."""
        best_x = None
        best_acquisition = float('-inf')
        
        # Try multiple starting points
        for _ in range(n_restarts):
            # Random starting point
            x0 = np.array([
                np.random.uniform(low, high)
                for low, high in self.bounds
            ])
            
            # Minimize negative acquisition function
            result = minimize(
                lambda x: -self._compute_acquisition(x),
                x0,
                bounds=self.bounds,
                method='L-BFGS-B'
            )
            
            if result.success:
                acquisition_value = -result.fun
                
                if acquisition_value > best_acquisition:
                    best_acquisition = acquisition_value
                    best_x = result.x
                    
        # Fallback to random point if optimization fails
        if best_x is None:
            best_x = np.array([
                np.random.uniform(low, high)
                for low, high in self.bounds
            ])
            
        return best_x
        
    def optimize(
        self,
        n_calls: int = 100,
        n_initial_points: Optional[int] = None,
        callback: Optional[Callable] = None
    ) -> BayesianOptimizationResult:
        """Run Bayesian optimization.
        
        Args:
            n_calls: Number of function evaluations
            n_initial_points: Override initial points
            callback: Callback function called after each evaluation
            
        Returns:
            Optimization results
        """
        if n_initial_points is not None:
            self.n_initial_points = n_initial_points
            
        logger.info(f"Starting Bayesian optimization with {n_calls} evaluations")
        
        # Phase 1: Initial random sampling
        initial_points = self._sample_initial_points()
        
        for i, x in enumerate(initial_points):
            score = self._evaluate_point(x)
            params = self._decode_params(x)
            
            self.X_observed.append(x)
            self.y_observed.append(score)
            
            # Track evaluation
            evaluation = {
                'iteration': i,
                'params': params,
                'score': score,
                'acquisition': 0.0  # No acquisition for initial points
            }
            self.evaluations.append(evaluation)
            
            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                
            # Callback
            if callback:
                callback(evaluation)
                
            logger.info(f"Initial point {i+1}/{len(initial_points)}: score={score:.6f}")
            
        # Phase 2: Bayesian optimization
        for iteration in range(self.n_initial_points, n_calls):
            # Fit Gaussian Process
            X_array = np.array(self.X_observed)
            y_array = np.array(self.y_observed)
            
            try:
                self.gp.fit(X_array, y_array)
            except Exception as e:
                logger.error(f"GP fitting failed: {str(e)}")
                # Fallback to random sampling
                x_next = np.array([
                    np.random.uniform(low, high)
                    for low, high in self.bounds
                ])
            else:
                # Optimize acquisition function
                x_next = self._optimize_acquisition()
                
            # Evaluate next point
            score = self._evaluate_point(x_next)
            params = self._decode_params(x_next)
            
            # Calculate acquisition value
            acquisition_value = self._compute_acquisition(x_next)
            
            # Update observations
            self.X_observed.append(x_next)
            self.y_observed.append(score)
            
            # Track evaluation
            evaluation = {
                'iteration': iteration,
                'params': params,
                'score': score,
                'acquisition': acquisition_value
            }
            self.evaluations.append(evaluation)
            
            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                logger.info(f"New best: score={score:.6f}")
                
            # Callback
            if callback:
                callback(evaluation)
                
            # Log progress
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: best={self.best_score:.6f}")
                
        # Create convergence history
        convergence_history = []
        current_best = float('-inf')
        
        for eval_data in self.evaluations:
            current_best = max(current_best, eval_data['score'])
            convergence_history.append(current_best)
            
        # Extract acquisition values
        acquisition_values = [eval_data['acquisition'] for eval_data in self.evaluations]
        
        logger.info(f"Optimization completed. Best score: {self.best_score:.6f}")
        
        return BayesianOptimizationResult(
            best_params=self.best_params,
            best_score=self.best_score,
            all_evaluations=self.evaluations,
            gp_model=deepcopy(self.gp),
            acquisition_values=acquisition_values,
            convergence_history=convergence_history,
            n_iterations=len(self.evaluations)
        )
        
    def suggest_next_point(self) -> Dict[str, Any]:
        """Suggest next point to evaluate."""
        if len(self.X_observed) < self.n_initial_points:
            # Still in initial sampling phase
            initial_points = self._sample_initial_points()
            x_next = initial_points[len(self.X_observed)]
        else:
            # Fit GP and optimize acquisition
            X_array = np.array(self.X_observed)
            y_array = np.array(self.y_observed)
            
            self.gp.fit(X_array, y_array)
            x_next = self._optimize_acquisition()
            
        return self._decode_params(x_next)
        
    def update(self, params: Dict[str, Any], score: float):
        """Update optimization with new evaluation."""
        x = self._encode_params(params)
        
        self.X_observed.append(x)
        self.y_observed.append(score)
        
        # Update best
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
            
    def get_parameter_importance(self) -> Dict[str, float]:
        """Get parameter importance based on GP lengthscales."""
        if not hasattr(self.gp, 'kernel_') or len(self.X_observed) == 0:
            return {}
            
        try:
            # Extract lengthscales from fitted kernel
            kernel = self.gp.kernel_
            
            if hasattr(kernel, 'k2') and hasattr(kernel.k2, 'length_scale'):
                # For ConstantKernel * Matern
                lengthscales = kernel.k2.length_scale
            elif hasattr(kernel, 'length_scale'):
                # Direct lengthscale
                lengthscales = kernel.length_scale
            else:
                return {}
                
            # Convert to importance (inverse of lengthscale)
            if np.isscalar(lengthscales):
                lengthscales = np.array([lengthscales] * self.n_dims)
                
            importance = 1.0 / (lengthscales + 1e-10)
            importance = importance / importance.sum()  # Normalize
            
            return {
                param_name: float(importance[i])
                for i, param_name in enumerate(self.param_names)
            }
            
        except Exception as e:
            logger.error(f"Error calculating parameter importance: {str(e)}")
            return {}
            
    def predict(self, params: Dict[str, Any]) -> Tuple[float, float]:
        """Predict mean and uncertainty for parameters."""
        if len(self.X_observed) == 0:
            return 0.0, 1.0
            
        x = self._encode_params(params)
        mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
        
        return float(mu[0]), float(sigma[0])
        
    def get_acquisition_surface(
        self,
        param1: str,
        param2: str,
        resolution: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get acquisition function surface for visualization."""
        if param1 not in self.param_names or param2 not in self.param_names:
            raise ValueError("Parameters not found")
            
        idx1 = self.param_names.index(param1)
        idx2 = self.param_names.index(param2)
        
        # Create grid
        bounds1 = self.bounds[idx1]
        bounds2 = self.bounds[idx2]
        
        x1 = np.linspace(bounds1[0], bounds1[1], resolution)
        x2 = np.linspace(bounds2[0], bounds2[1], resolution)
        X1, X2 = np.meshgrid(x1, x2)
        
        # Evaluate acquisition function
        Z = np.zeros_like(X1)
        
        # Use mean values for other parameters
        mean_params = {
            name: (bounds[0] + bounds[1]) / 2
            for name, bounds in zip(self.param_names, self.bounds)
        }
        
        for i in range(resolution):
            for j in range(resolution):
                x = np.array([mean_params[name] for name in self.param_names])
                x[idx1] = X1[i, j]
                x[idx2] = X2[i, j]
                
                Z[i, j] = self._compute_acquisition(x)
                
        return X1, X2, Z