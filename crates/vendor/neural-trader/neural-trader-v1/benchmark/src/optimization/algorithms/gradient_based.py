"""Gradient-based optimization methods for parameter tuning."""

import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
import logging
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
import time

logger = logging.getLogger(__name__)


@dataclass
class GradientOptimizationResult:
    """Result from gradient-based optimization."""
    best_params: Dict[str, Any]
    best_score: float
    optimization_path: List[Dict[str, Any]]
    convergence_curve: List[float]
    n_evaluations: int
    optimization_time: float
    success: bool
    method_used: str


class GradientOptimizer:
    """Gradient-based optimization methods."""
    
    def __init__(
        self,
        objective_function: Callable,
        parameter_space: Dict[str, Any],
        gradient_function: Optional[Callable] = None,
        finite_diff_step: float = 1e-6
    ):
        """Initialize gradient optimizer.
        
        Args:
            objective_function: Function to optimize (will be minimized)
            parameter_space: Dictionary defining parameter bounds
            gradient_function: Optional gradient function
            finite_diff_step: Step size for finite differences
        """
        self.objective_function = objective_function
        self.parameter_space = parameter_space
        self.gradient_function = gradient_function
        self.finite_diff_step = finite_diff_step
        
        # Process parameter space
        self.param_names = list(parameter_space.keys())
        self.bounds = self._process_parameter_space()
        
        self.evaluation_history = []
        self.n_evaluations = 0
        
    def _process_parameter_space(self) -> List[Tuple[float, float]]:
        """Process parameter space into bounds."""
        bounds = []
        
        for param_name in self.param_names:
            param_def = self.parameter_space[param_name]
            
            if isinstance(param_def, tuple) and len(param_def) == 2:
                bounds.append(param_def)
            elif isinstance(param_def, dict):
                low = param_def.get('low', 0.0)
                high = param_def.get('high', 1.0)
                bounds.append((low, high))
            else:
                # Single value - create small range around it
                value = float(param_def)
                bounds.append((value - 0.1, value + 0.1))
                
        return bounds
        
    def _encode_params(self, params: Dict[str, Any]) -> np.ndarray:
        """Encode parameters to array."""
        return np.array([params[name] for name in self.param_names])
        
    def _decode_params(self, x: np.ndarray) -> Dict[str, Any]:
        """Decode array to parameters."""
        return {name: float(x[i]) for i, name in enumerate(self.param_names)}
        
    def _objective_wrapper(self, x: np.ndarray) -> float:
        """Wrapper for objective function (for minimization)."""
        params = self._decode_params(x)
        
        try:
            # Negate for minimization (assuming we want to maximize)
            score = -float(self.objective_function(params))
            
            # Track evaluation
            self.evaluation_history.append({
                'params': params,
                'score': -score,  # Store original (maximization) score
                'evaluation': self.n_evaluations
            })
            self.n_evaluations += 1
            
            return score
            
        except Exception as e:
            logger.error(f"Error evaluating {params}: {str(e)}")
            return float('inf')  # Return high value for minimization
            
    def _gradient_wrapper(self, x: np.ndarray) -> np.ndarray:
        """Wrapper for gradient function."""
        if self.gradient_function is None:
            return self._finite_difference_gradient(x)
        else:
            params = self._decode_params(x)
            grad_dict = self.gradient_function(params)
            return -np.array([grad_dict[name] for name in self.param_names])
            
    def _finite_difference_gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient using finite differences."""
        gradient = np.zeros_like(x)
        f0 = self._objective_wrapper(x)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += self.finite_diff_step
            
            # Check bounds
            if x_plus[i] > self.bounds[i][1]:
                x_plus[i] = self.bounds[i][1]
                
            f_plus = self._objective_wrapper(x_plus)
            gradient[i] = (f_plus - f0) / self.finite_diff_step
            
        return gradient
        
    def optimize_bfgs(
        self,
        initial_params: Optional[Dict[str, Any]] = None,
        max_iterations: int = 1000,
        tolerance: float = 1e-6
    ) -> GradientOptimizationResult:
        """Optimize using L-BFGS-B algorithm."""
        start_time = time.time()
        
        # Initial point
        if initial_params is None:
            x0 = np.array([(b[0] + b[1]) / 2 for b in self.bounds])
        else:
            x0 = self._encode_params(initial_params)
            
        logger.info("Starting L-BFGS-B optimization")
        
        # Optimize
        result = minimize(
            fun=self._objective_wrapper,
            x0=x0,
            method='L-BFGS-B',
            bounds=self.bounds,
            jac=self._gradient_wrapper if self.gradient_function else None,
            options={
                'maxiter': max_iterations,
                'ftol': tolerance,
                'gtol': tolerance
            }
        )
        
        optimization_time = time.time() - start_time
        
        # Best parameters
        best_params = self._decode_params(result.x)
        best_score = -result.fun  # Convert back to maximization
        
        # Convergence curve
        convergence_curve = []
        current_best = float('-inf')
        for eval_data in self.evaluation_history:
            current_best = max(current_best, eval_data['score'])
            convergence_curve.append(current_best)
            
        return GradientOptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_path=self.evaluation_history,
            convergence_curve=convergence_curve,
            n_evaluations=self.n_evaluations,
            optimization_time=optimization_time,
            success=result.success,
            method_used='L-BFGS-B'
        )
        
    def optimize_nelder_mead(
        self,
        initial_params: Optional[Dict[str, Any]] = None,
        max_iterations: int = 1000,
        tolerance: float = 1e-6
    ) -> GradientOptimizationResult:
        """Optimize using Nelder-Mead simplex algorithm."""
        start_time = time.time()
        
        # Initial point
        if initial_params is None:
            x0 = np.array([(b[0] + b[1]) / 2 for b in self.bounds])
        else:
            x0 = self._encode_params(initial_params)
            
        logger.info("Starting Nelder-Mead optimization")
        
        # Optimize
        result = minimize(
            fun=self._objective_wrapper,
            x0=x0,
            method='Nelder-Mead',
            options={
                'maxiter': max_iterations,
                'xatol': tolerance,
                'fatol': tolerance
            }
        )
        
        optimization_time = time.time() - start_time
        
        # Best parameters
        best_params = self._decode_params(result.x)
        best_score = -result.fun
        
        # Convergence curve
        convergence_curve = []
        current_best = float('-inf')
        for eval_data in self.evaluation_history:
            current_best = max(current_best, eval_data['score'])
            convergence_curve.append(current_best)
            
        return GradientOptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_path=self.evaluation_history,
            convergence_curve=convergence_curve,
            n_evaluations=self.n_evaluations,
            optimization_time=optimization_time,
            success=result.success,
            method_used='Nelder-Mead'
        )
        
    def optimize_differential_evolution(
        self,
        population_size: int = 15,
        max_generations: int = 1000,
        tolerance: float = 1e-6,
        mutation_rate: float = 0.5,
        crossover_rate: float = 0.7
    ) -> GradientOptimizationResult:
        """Optimize using Differential Evolution."""
        start_time = time.time()
        
        logger.info("Starting Differential Evolution optimization")
        
        # Optimize
        result = differential_evolution(
            func=self._objective_wrapper,
            bounds=self.bounds,
            popsize=population_size,
            maxiter=max_generations,
            tol=tolerance,
            mutation=mutation_rate,
            recombination=crossover_rate,
            seed=42
        )
        
        optimization_time = time.time() - start_time
        
        # Best parameters
        best_params = self._decode_params(result.x)
        best_score = -result.fun
        
        # Convergence curve
        convergence_curve = []
        current_best = float('-inf')
        for eval_data in self.evaluation_history:
            current_best = max(current_best, eval_data['score'])
            convergence_curve.append(current_best)
            
        return GradientOptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_path=self.evaluation_history,
            convergence_curve=convergence_curve,
            n_evaluations=self.n_evaluations,
            optimization_time=optimization_time,
            success=result.success,
            method_used='Differential Evolution'
        )
        
    def optimize_trust_region(
        self,
        initial_params: Optional[Dict[str, Any]] = None,
        max_iterations: int = 1000,
        tolerance: float = 1e-6
    ) -> GradientOptimizationResult:
        """Optimize using Trust Region algorithm."""
        start_time = time.time()
        
        # Initial point
        if initial_params is None:
            x0 = np.array([(b[0] + b[1]) / 2 for b in self.bounds])
        else:
            x0 = self._encode_params(initial_params)
            
        logger.info("Starting Trust Region optimization")
        
        # Convert bounds to constraints
        from scipy.optimize import Bounds
        bounds_obj = Bounds(
            lb=[b[0] for b in self.bounds],
            ub=[b[1] for b in self.bounds]
        )
        
        # Optimize
        result = minimize(
            fun=self._objective_wrapper,
            x0=x0,
            method='trust-constr',
            jac=self._gradient_wrapper if self.gradient_function else None,
            bounds=bounds_obj,
            options={
                'maxiter': max_iterations,
                'gtol': tolerance,
                'xtol': tolerance
            }
        )
        
        optimization_time = time.time() - start_time
        
        # Best parameters
        best_params = self._decode_params(result.x)
        best_score = -result.fun
        
        # Convergence curve
        convergence_curve = []
        current_best = float('-inf')
        for eval_data in self.evaluation_history:
            current_best = max(current_best, eval_data['score'])
            convergence_curve.append(current_best)
            
        return GradientOptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_path=self.evaluation_history,
            convergence_curve=convergence_curve,
            n_evaluations=self.n_evaluations,
            optimization_time=optimization_time,
            success=result.success,
            method_used='Trust Region'
        )
        
    def optimize(
        self,
        method: str = 'L-BFGS-B',
        initial_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> GradientOptimizationResult:
        """Run optimization with specified method.
        
        Args:
            method: Optimization method ('L-BFGS-B', 'Nelder-Mead', 'DE', 'Trust-Region')
            initial_params: Initial parameter values
            **kwargs: Method-specific parameters
            
        Returns:
            Optimization results
        """
        # Reset history
        self.evaluation_history = []
        self.n_evaluations = 0
        
        if method.upper() == 'L-BFGS-B':
            return self.optimize_bfgs(initial_params, **kwargs)
        elif method.upper() == 'NELDER-MEAD':
            return self.optimize_nelder_mead(initial_params, **kwargs)
        elif method.upper() == 'DE':
            return self.optimize_differential_evolution(**kwargs)
        elif method.upper() == 'TRUST-REGION':
            return self.optimize_trust_region(initial_params, **kwargs)
        else:
            raise ValueError(f"Unknown optimization method: {method}")


class AdamOptimizer:
    """Adam optimizer for gradient-based optimization."""
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        """Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            epsilon: Small constant for numerical stability
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step
        
    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Perform one optimization step.
        
        Args:
            params: Current parameters
            gradients: Parameter gradients
            
        Returns:
            Updated parameters
        """
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            
        self.t += 1
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        
        # Update biased second moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Update parameters
        updated_params = params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return updated_params