"""
Convergence analysis tools for optimization algorithms.

Analyzes convergence behavior, detects plateaus, and predicts
convergence for optimization runs.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from scipy.optimize import curve_fit
import warnings
from dataclasses import dataclass

from .parameter_optimizer import OptimizationResult

warnings.filterwarnings('ignore')


@dataclass
class ConvergenceAnalysis:
    """Results of convergence analysis."""
    convergence_rate: float
    convergence_type: str
    r_squared: float
    parameters: Dict[str, float]
    predicted_limit: Optional[float] = None


@dataclass
class Plateau:
    """Detected plateau in optimization."""
    start: int
    end: int
    duration: int
    mean_value: float
    std_value: float


@dataclass
class ConvergencePrediction:
    """Convergence prediction results."""
    will_converge: bool
    estimated_iterations: Optional[int]
    confidence_interval: Optional[Tuple[int, int]]
    predicted_value: Optional[float]
    confidence: float


class ConvergenceAnalyzer:
    """Analyzes convergence behavior of optimization algorithms."""
    
    def __init__(self):
        """Initialize convergence analyzer."""
        self.convergence_models = {
            'exponential': self._exponential_model,
            'power_law': self._power_law_model,
            'logarithmic': self._logarithmic_model,
            'linear': self._linear_model
        }
    
    def analyze(self, result: OptimizationResult) -> Dict[str, Any]:
        """
        Analyze convergence behavior of optimization run.
        
        Args:
            result: Optimization result to analyze
            
        Returns:
            Dictionary with convergence analysis
        """
        scores = [h['score'] for h in result.history]
        iterations = np.arange(len(scores))
        
        # Try different convergence models
        best_model = None
        best_r2 = -float('inf')
        best_params = None
        
        for model_name, model_func in self.convergence_models.items():
            try:
                params, r2 = self._fit_model(iterations, scores, model_func)
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model_name
                    best_params = params
            except:
                continue
        
        # Calculate convergence rate
        if best_model == 'exponential' and best_params:
            # Rate is the decay parameter
            convergence_rate = abs(best_params.get('b', 0))
        elif best_model == 'power_law' and best_params:
            # Rate is the power parameter
            convergence_rate = abs(best_params.get('b', 0))
        else:
            # Approximate rate from improvement
            if len(scores) > 10:
                early_score = np.mean(scores[:10])
                late_score = np.mean(scores[-10:])
                convergence_rate = abs(late_score - early_score) / len(scores)
            else:
                convergence_rate = 0.0
        
        # Predict limit value
        if best_model and best_params:
            if best_model == 'exponential':
                predicted_limit = best_params.get('c', scores[-1])
            elif best_model == 'logarithmic':
                # Logarithmic doesn't have a limit
                predicted_limit = None
            else:
                # Extrapolate
                future_iteration = len(scores) * 2
                predicted_limit = self.convergence_models[best_model](
                    future_iteration, **best_params
                )
        else:
            predicted_limit = None
        
        return {
            'convergence_rate': convergence_rate,
            'convergence_type': best_model or 'unknown',
            'r_squared': best_r2,
            'parameters': best_params or {},
            'predicted_limit': predicted_limit,
            'is_converged': result.converged,
            'convergence_iteration': result.convergence_iteration
        }
    
    def detect_plateaus(self, result: OptimizationResult, 
                       window_size: int = 10,
                       threshold: float = 0.01) -> List[Plateau]:
        """
        Detect plateaus in optimization history.
        
        Args:
            result: Optimization result
            window_size: Window size for plateau detection
            threshold: Threshold for considering values as plateau
            
        Returns:
            List of detected plateaus
        """
        scores = [h['score'] for h in result.history]
        plateaus = []
        
        i = 0
        while i < len(scores) - window_size:
            window = scores[i:i + window_size]
            window_std = np.std(window)
            window_mean = np.mean(window)
            
            if window_std < threshold * abs(window_mean):
                # Potential plateau start
                plateau_start = i
                
                # Extend plateau
                j = i + window_size
                while j < len(scores):
                    extended_window = scores[plateau_start:j+1]
                    if np.std(extended_window) < threshold * abs(np.mean(extended_window)):
                        j += 1
                    else:
                        break
                
                plateau_end = j - 1
                duration = plateau_end - plateau_start + 1
                
                if duration >= window_size:
                    plateau = Plateau(
                        start=plateau_start,
                        end=plateau_end,
                        duration=duration,
                        mean_value=np.mean(scores[plateau_start:plateau_end+1]),
                        std_value=np.std(scores[plateau_start:plateau_end+1])
                    )
                    plateaus.append(plateau)
                    i = plateau_end + 1
                else:
                    i += 1
            else:
                i += 1
        
        return plateaus
    
    def predict_convergence(self, partial_result: OptimizationResult,
                           target_score: float,
                           confidence_level: float = 0.95) -> ConvergencePrediction:
        """
        Predict if and when optimization will converge.
        
        Args:
            partial_result: Partial optimization result
            target_score: Target score to reach
            confidence_level: Confidence level for prediction
            
        Returns:
            Convergence prediction
        """
        scores = [h['score'] for h in partial_result.history]
        iterations = np.arange(len(scores))
        
        if len(scores) < 10:
            return ConvergencePrediction(
                will_converge=False,
                estimated_iterations=None,
                confidence_interval=None,
                predicted_value=None,
                confidence=0.0
            )
        
        # Fit best model
        analysis = self.analyze(partial_result)
        
        if analysis['convergence_type'] == 'unknown':
            return ConvergencePrediction(
                will_converge=False,
                estimated_iterations=None,
                confidence_interval=None,
                predicted_value=None,
                confidence=0.0
            )
        
        model_func = self.convergence_models[analysis['convergence_type']]
        params = analysis['parameters']
        
        # Check if already below target
        if scores[-1] <= target_score:
            return ConvergencePrediction(
                will_converge=True,
                estimated_iterations=len(scores),
                confidence_interval=(len(scores), len(scores)),
                predicted_value=scores[-1],
                confidence=1.0
            )
        
        # Predict future values
        will_converge = False
        estimated_iterations = None
        
        # Search for convergence point
        for future_iter in range(len(scores), len(scores) * 100):
            try:
                predicted_value = model_func(future_iter, **params)
                if predicted_value <= target_score:
                    will_converge = True
                    estimated_iterations = future_iter
                    break
            except:
                break
        
        # Calculate confidence based on model fit
        confidence = min(analysis['r_squared'], 0.99)
        
        # Estimate confidence interval
        if will_converge and estimated_iterations:
            # Simple confidence interval based on uncertainty
            uncertainty = (1 - confidence) * estimated_iterations
            lower_bound = max(len(scores), int(estimated_iterations - uncertainty))
            upper_bound = int(estimated_iterations + uncertainty)
            confidence_interval = (lower_bound, upper_bound)
        else:
            confidence_interval = None
        
        return ConvergencePrediction(
            will_converge=will_converge,
            estimated_iterations=estimated_iterations,
            confidence_interval=confidence_interval,
            predicted_value=target_score if will_converge else None,
            confidence=confidence
        )
    
    def multi_run_statistics(self, results: List[OptimizationResult]) -> Dict[str, Any]:
        """
        Analyze convergence statistics across multiple runs.
        
        Args:
            results: List of optimization results
            
        Returns:
            Statistics dictionary
        """
        convergence_iterations = []
        final_scores = []
        convergence_rates = []
        convergence_types = []
        
        for result in results:
            # Analyze each run
            analysis = self.analyze(result)
            
            if result.converged and result.convergence_iteration:
                convergence_iterations.append(result.convergence_iteration)
            
            final_scores.append(result.best_score)
            convergence_rates.append(analysis['convergence_rate'])
            convergence_types.append(analysis['convergence_type'])
        
        # Calculate statistics
        stats = {
            'n_runs': len(results),
            'success_rate': sum(1 for r in results if r.converged) / len(results),
            'mean_final_score': np.mean(final_scores),
            'std_final_score': np.std(final_scores),
            'min_final_score': np.min(final_scores),
            'max_final_score': np.max(final_scores)
        }
        
        if convergence_iterations:
            stats.update({
                'mean_convergence_iteration': np.mean(convergence_iterations),
                'std_convergence_iteration': np.std(convergence_iterations),
                'min_convergence_iteration': np.min(convergence_iterations),
                'max_convergence_iteration': np.max(convergence_iterations)
            })
        
        # Convergence rate statistics
        if convergence_rates:
            stats.update({
                'mean_convergence_rate': np.mean(convergence_rates),
                'std_convergence_rate': np.std(convergence_rates)
            })
        
        # Most common convergence type
        if convergence_types:
            type_counts = {}
            for ct in convergence_types:
                type_counts[ct] = type_counts.get(ct, 0) + 1
            most_common_type = max(type_counts, key=type_counts.get)
            stats['most_common_convergence_type'] = most_common_type
            stats['convergence_type_distribution'] = type_counts
        
        return stats
    
    # Model fitting functions
    def _exponential_model(self, x, a, b, c):
        """Exponential decay model: a * exp(-b * x) + c"""
        return a * np.exp(-b * x) + c
    
    def _power_law_model(self, x, a, b, c):
        """Power law model: a * x^(-b) + c"""
        return a * np.power(x + 1, -b) + c
    
    def _logarithmic_model(self, x, a, b):
        """Logarithmic model: a * log(x + 1) + b"""
        return a * np.log(x + 1) + b
    
    def _linear_model(self, x, a, b):
        """Linear model: a * x + b"""
        return a * x + b
    
    def _fit_model(self, x, y, model_func):
        """Fit a model to data and return parameters and R-squared."""
        try:
            # Initial parameter guess
            if model_func == self._exponential_model:
                p0 = [y[0] - y[-1], 0.01, y[-1]]
                bounds = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])
            elif model_func == self._power_law_model:
                p0 = [y[0] - y[-1], 0.5, y[-1]]
                bounds = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])
            elif model_func == self._logarithmic_model:
                p0 = [(y[-1] - y[0]) / np.log(len(y)), y[0]]
                bounds = None
            else:  # linear
                p0 = [(y[-1] - y[0]) / len(y), y[0]]
                bounds = None
            
            # Fit model
            if bounds:
                params, _ = curve_fit(model_func, x, y, p0=p0, bounds=bounds, maxfev=5000)
            else:
                params, _ = curve_fit(model_func, x, y, p0=p0, maxfev=5000)
            
            # Calculate R-squared
            y_pred = model_func(x, *params)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Return as dictionary
            if model_func == self._exponential_model:
                param_dict = {'a': params[0], 'b': params[1], 'c': params[2]}
            elif model_func == self._power_law_model:
                param_dict = {'a': params[0], 'b': params[1], 'c': params[2]}
            elif model_func == self._logarithmic_model:
                param_dict = {'a': params[0], 'b': params[1]}
            else:  # linear
                param_dict = {'a': params[0], 'b': params[1]}
            
            return param_dict, r_squared
            
        except Exception as e:
            return {}, -1.0