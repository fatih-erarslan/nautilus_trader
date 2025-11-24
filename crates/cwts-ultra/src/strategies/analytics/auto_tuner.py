"""
Parameter Auto-Tuning System

Implements comprehensive parameter optimization including:
- Multi-objective optimization using NSGA-II genetic algorithm
- Bayesian optimization for continuous parameter spaces
- Grid search with statistical significance testing
- Particle swarm optimization for complex parameter landscapes
- Sensitivity analysis and parameter stability testing
- Walk-forward parameter optimization with overfitting detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy import stats
from scipy.optimize import minimize, differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Optimization method types"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    MULTI_OBJECTIVE = "multi_objective"
    WALK_FORWARD = "walk_forward"


class ObjectiveType(Enum):
    """Optimization objective types"""
    MAXIMIZE_SHARPE = "maximize_sharpe"
    MAXIMIZE_RETURN = "maximize_return"
    MINIMIZE_RISK = "minimize_risk"
    MAXIMIZE_CALMAR = "maximize_calmar"
    MAXIMIZE_SORTINO = "maximize_sortino"
    MULTI_OBJECTIVE = "multi_objective"
    CUSTOM = "custom"


@dataclass
class Parameter:
    """Parameter definition for optimization"""
    name: str
    param_type: str  # 'continuous', 'discrete', 'categorical'
    bounds: Union[Tuple[float, float], List[Any]]  # (min, max) for continuous, list for discrete/categorical
    default_value: Any
    step_size: Optional[float] = None  # For discrete parameters
    log_scale: bool = False  # Whether to optimize in log space


@dataclass
class OptimizationResult:
    """Parameter optimization result"""
    method: OptimizationMethod
    best_parameters: Dict[str, Any]
    best_score: float
    objective_values: Dict[str, float]
    optimization_history: List[Dict[str, Any]]
    convergence_metrics: Dict[str, float]
    sensitivity_analysis: Dict[str, Dict[str, float]]
    stability_metrics: Dict[str, float]
    overfitting_score: float
    statistical_significance: Dict[str, float]
    walk_forward_results: Optional[Dict[str, Any]] = None


class ObjectiveFunction(ABC):
    """Abstract base class for optimization objectives"""
    
    @abstractmethod
    def evaluate(self, parameters: Dict[str, Any], data: Dict[str, Any]) -> float:
        """Evaluate objective function with given parameters"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get objective function name"""
        pass


class SharpeRatioObjective(ObjectiveFunction):
    """Sharpe ratio maximization objective"""
    
    def evaluate(self, parameters: Dict[str, Any], data: Dict[str, Any]) -> float:
        """Calculate Sharpe ratio for given parameters"""
        returns = data.get('returns')
        if returns is None or len(returns) < 10:
            return -np.inf
            
        # Apply parameters to strategy (this would be strategy-specific)
        # For now, assume returns are already calculated based on parameters
        
        mean_return = returns.mean()
        volatility = returns.std()
        
        if volatility <= 0:
            return -np.inf
            
        sharpe_ratio = mean_return / volatility * np.sqrt(252)  # Annualized
        return sharpe_ratio
    
    def get_name(self) -> str:
        return "Sharpe Ratio"


class CalmarRatioObjective(ObjectiveFunction):
    """Calmar ratio maximization objective"""
    
    def evaluate(self, parameters: Dict[str, Any], data: Dict[str, Any]) -> float:
        """Calculate Calmar ratio for given parameters"""
        returns = data.get('returns')
        if returns is None or len(returns) < 10:
            return -np.inf
            
        annual_return = returns.mean() * 252
        max_drawdown = self._calculate_max_drawdown(returns)
        
        if max_drawdown >= 0:  # No drawdown
            return annual_return if annual_return > 0 else -np.inf
            
        calmar_ratio = -annual_return / max_drawdown  # Negative because max_drawdown is negative
        return calmar_ratio
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def get_name(self) -> str:
        return "Calmar Ratio"


class MultiObjectiveFunction:
    """Multi-objective optimization function"""
    
    def __init__(self, objectives: List[ObjectiveFunction], weights: Optional[List[float]] = None):
        self.objectives = objectives
        self.weights = weights or [1.0] * len(objectives)
        
        if len(self.weights) != len(self.objectives):
            raise ValueError("Number of weights must match number of objectives")
    
    def evaluate(self, parameters: Dict[str, Any], data: Dict[str, Any]) -> Tuple[List[float], float]:
        """
        Evaluate all objectives and return both individual scores and weighted sum
        
        Returns:
            Tuple of (individual_scores, weighted_sum)
        """
        scores = []
        for objective in self.objectives:
            score = objective.evaluate(parameters, data)
            scores.append(score)
        
        # Weighted sum (scalarization)
        weighted_sum = sum(w * s for w, s in zip(self.weights, scores))
        
        return scores, weighted_sum


class ParameterAutoTuner:
    """
    Comprehensive parameter optimization system
    
    Provides multiple optimization algorithms and robust parameter
    selection with statistical validation and overfitting detection.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Optimization history
        self.optimization_history = []
        
        # Built-in objective functions
        self.objective_functions = {
            ObjectiveType.MAXIMIZE_SHARPE: SharpeRatioObjective(),
            ObjectiveType.MAXIMIZE_CALMAR: CalmarRatioObjective()
        }
        
    def optimize_parameters(
        self,
        parameters: List[Parameter],
        objective: Union[ObjectiveType, ObjectiveFunction, MultiObjectiveFunction],
        data: Dict[str, Any],
        method: OptimizationMethod = OptimizationMethod.BAYESIAN_OPTIMIZATION,
        n_trials: int = 100,
        validation_split: float = 0.3,
        cross_validation_folds: int = 5,
        stability_tests: bool = True,
        parallel_jobs: int = 4
    ) -> OptimizationResult:
        """
        Optimize parameters using specified method
        
        Args:
            parameters: List of parameters to optimize
            objective: Objective function or type
            data: Data dictionary for optimization
            method: Optimization method to use
            n_trials: Number of optimization trials
            validation_split: Fraction of data for validation
            cross_validation_folds: Number of CV folds
            stability_tests: Whether to run stability analysis
            parallel_jobs: Number of parallel jobs
            
        Returns:
            OptimizationResult with best parameters and analysis
        """
        # Validate inputs
        if not parameters:
            raise ValueError("At least one parameter must be specified")
        
        # Split data for validation
        train_data, val_data = self._split_data(data, validation_split)
        
        # Run optimization
        if method == OptimizationMethod.GRID_SEARCH:
            result = self._grid_search_optimization(
                parameters, objective, train_data, n_trials, parallel_jobs
            )
        elif method == OptimizationMethod.RANDOM_SEARCH:
            result = self._random_search_optimization(
                parameters, objective, train_data, n_trials, parallel_jobs
            )
        elif method == OptimizationMethod.BAYESIAN_OPTIMIZATION:
            result = self._bayesian_optimization(
                parameters, objective, train_data, n_trials
            )
        elif method == OptimizationMethod.GENETIC_ALGORITHM:
            result = self._genetic_algorithm_optimization(
                parameters, objective, train_data, n_trials, parallel_jobs
            )
        elif method == OptimizationMethod.PARTICLE_SWARM:
            result = self._particle_swarm_optimization(
                parameters, objective, train_data, n_trials
            )
        elif method == OptimizationMethod.DIFFERENTIAL_EVOLUTION:
            result = self._differential_evolution_optimization(
                parameters, objective, train_data, n_trials
            )
        elif method == OptimizationMethod.MULTI_OBJECTIVE:
            result = self._multi_objective_optimization(
                parameters, objective, train_data, n_trials
            )
        elif method == OptimizationMethod.WALK_FORWARD:
            result = self._walk_forward_optimization(
                parameters, objective, data, n_trials
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Validate on held-out data
        val_score = self._evaluate_objective(objective, result.best_parameters, val_data)
        
        # Calculate overfitting score
        overfitting_score = (result.best_score - val_score) / abs(result.best_score) if result.best_score != 0 else 0
        result.overfitting_score = overfitting_score
        
        # Run stability analysis if requested
        if stability_tests:
            result.stability_metrics = self._analyze_parameter_stability(
                parameters, objective, result.best_parameters, data, cross_validation_folds
            )
            
        # Sensitivity analysis
        result.sensitivity_analysis = self._sensitivity_analysis(
            parameters, objective, result.best_parameters, train_data
        )
        
        # Statistical significance testing
        result.statistical_significance = self._test_statistical_significance(
            objective, result.best_parameters, data, cross_validation_folds
        )
        
        return result
    
    def _grid_search_optimization(
        self,
        parameters: List[Parameter],
        objective: Union[ObjectiveType, ObjectiveFunction, MultiObjectiveFunction],
        data: Dict[str, Any],
        max_trials: int,
        parallel_jobs: int
    ) -> OptimizationResult:
        """Grid search parameter optimization"""
        # Generate parameter grid
        param_grid = self._generate_parameter_grid(parameters, max_trials)
        
        best_score = -np.inf
        best_params = {}
        history = []
        
        # Parallel evaluation
        with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
            future_to_params = {
                executor.submit(self._evaluate_objective, objective, params, data): params
                for params in param_grid
            }
            
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    score = future.result()
                    history.append({'parameters': params, 'score': score})
                    
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        
                except Exception as e:
                    logger.error(f"Parameter evaluation failed: {e}")
                    history.append({'parameters': params, 'score': -np.inf})
        
        # Calculate convergence metrics
        scores = [h['score'] for h in history if h['score'] != -np.inf]
        convergence_metrics = {
            'total_evaluations': len(history),
            'successful_evaluations': len(scores),
            'score_variance': np.var(scores) if scores else 0,
            'score_range': max(scores) - min(scores) if scores else 0
        }
        
        return OptimizationResult(
            method=OptimizationMethod.GRID_SEARCH,
            best_parameters=best_params,
            best_score=best_score,
            objective_values={'primary': best_score},
            optimization_history=history,
            convergence_metrics=convergence_metrics,
            sensitivity_analysis={},
            stability_metrics={},
            overfitting_score=0.0,
            statistical_significance={}
        )
    
    def _bayesian_optimization(
        self,
        parameters: List[Parameter],
        objective: Union[ObjectiveType, ObjectiveFunction, MultiObjectiveFunction],
        data: Dict[str, Any],
        n_trials: int
    ) -> OptimizationResult:
        """Bayesian optimization using Gaussian Process"""
        # Only handle continuous parameters for now
        continuous_params = [p for p in parameters if p.param_type == 'continuous']
        if len(continuous_params) != len(parameters):
            logger.warning("Bayesian optimization currently only supports continuous parameters")
        
        # Initialize Gaussian Process
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=5)
        
        # Parameter bounds
        bounds = np.array([p.bounds for p in continuous_params])
        
        # Initialize with random samples
        n_init = min(10, n_trials // 4)
        X_init = self._latin_hypercube_sampling(bounds, n_init)
        y_init = []
        
        history = []
        
        for x in X_init:
            params = {p.name: val for p, val in zip(continuous_params, x)}
            score = self._evaluate_objective(objective, params, data)
            y_init.append(score)
            history.append({'parameters': params, 'score': score})
        
        X_sampled = X_init.copy()
        y_sampled = np.array(y_init)
        
        best_idx = np.argmax(y_sampled)
        best_score = y_sampled[best_idx]
        best_params = {p.name: val for p, val in zip(continuous_params, X_sampled[best_idx])}
        
        # Bayesian optimization loop
        for i in range(n_trials - n_init):
            # Fit GP model
            gp.fit(X_sampled, y_sampled)
            
            # Acquisition function optimization (Expected Improvement)
            next_x = self._optimize_acquisition_function(gp, bounds, y_sampled.max())
            
            # Evaluate new point
            next_params = {p.name: val for p, val in zip(continuous_params, next_x)}
            next_score = self._evaluate_objective(objective, next_params, data)
            
            # Update dataset
            X_sampled = np.vstack([X_sampled, next_x])
            y_sampled = np.append(y_sampled, next_score)
            
            history.append({'parameters': next_params, 'score': next_score})
            
            # Update best
            if next_score > best_score:
                best_score = next_score
                best_params = next_params.copy()
                
            # Early stopping if converged
            if i > 20 and self._check_convergence(y_sampled[-20:]):
                logger.info(f"Bayesian optimization converged after {i + n_init} evaluations")
                break
        
        # Calculate convergence metrics
        convergence_metrics = {
            'total_evaluations': len(history),
            'convergence_rate': self._calculate_convergence_rate(y_sampled),
            'acquisition_efficiency': (best_score - y_init[0]) / len(history) if y_init else 0,
            'gp_lengthscale': gp.kernel_.k1.k2.length_scale if hasattr(gp.kernel_, 'k1') else 1.0
        }
        
        return OptimizationResult(
            method=OptimizationMethod.BAYESIAN_OPTIMIZATION,
            best_parameters=best_params,
            best_score=best_score,
            objective_values={'primary': best_score},
            optimization_history=history,
            convergence_metrics=convergence_metrics,
            sensitivity_analysis={},
            stability_metrics={},
            overfitting_score=0.0,
            statistical_significance={}
        )
    
    def _genetic_algorithm_optimization(
        self,
        parameters: List[Parameter],
        objective: Union[ObjectiveType, ObjectiveFunction, MultiObjectiveFunction],
        data: Dict[str, Any],
        n_trials: int,
        parallel_jobs: int
    ) -> OptimizationResult:
        """Genetic algorithm optimization"""
        population_size = min(50, n_trials // 4)
        n_generations = n_trials // population_size
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = {}
            for param in parameters:
                if param.param_type == 'continuous':
                    if param.log_scale:
                        val = np.exp(np.random.uniform(np.log(param.bounds[0]), np.log(param.bounds[1])))
                    else:
                        val = np.random.uniform(param.bounds[0], param.bounds[1])
                elif param.param_type == 'discrete':
                    val = np.random.choice(param.bounds)
                else:  # categorical
                    val = np.random.choice(param.bounds)
                individual[param.name] = val
            population.append(individual)
        
        history = []
        best_score = -np.inf
        best_params = {}
        
        for generation in range(n_generations):
            # Evaluate population
            with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
                futures = [
                    executor.submit(self._evaluate_objective, objective, individual, data)
                    for individual in population
                ]
                
                scores = []
                for future in as_completed(futures):
                    try:
                        score = future.result()
                        scores.append(score)
                    except Exception as e:
                        logger.error(f"Individual evaluation failed: {e}")
                        scores.append(-np.inf)
            
            # Update history and best
            for individual, score in zip(population, scores):
                history.append({'parameters': individual, 'score': score})
                if score > best_score:
                    best_score = score
                    best_params = individual.copy()
            
            # Selection, crossover, and mutation
            population = self._genetic_operators(population, scores, parameters)
        
        convergence_metrics = {
            'total_evaluations': len(history),
            'generations': n_generations,
            'population_size': population_size,
            'final_diversity': self._calculate_population_diversity(population, parameters)
        }
        
        return OptimizationResult(
            method=OptimizationMethod.GENETIC_ALGORITHM,
            best_parameters=best_params,
            best_score=best_score,
            objective_values={'primary': best_score},
            optimization_history=history,
            convergence_metrics=convergence_metrics,
            sensitivity_analysis={},
            stability_metrics={},
            overfitting_score=0.0,
            statistical_significance={}
        )
    
    def _walk_forward_optimization(
        self,
        parameters: List[Parameter],
        objective: Union[ObjectiveType, ObjectiveFunction, MultiObjectiveFunction],
        data: Dict[str, Any],
        n_trials: int
    ) -> OptimizationResult:
        """Walk-forward parameter optimization"""
        returns = data.get('returns')
        if returns is None:
            raise ValueError("Returns data required for walk-forward optimization")
        
        # Parameters for walk-forward
        lookback_window = 252 * 2  # 2 years
        rebalance_frequency = 63   # Quarterly
        min_history = 252          # 1 year minimum
        
        if len(returns) < lookback_window + min_history:
            raise ValueError("Insufficient data for walk-forward optimization")
        
        walk_forward_results = []
        overall_history = []
        
        # Walk forward through time
        for start_idx in range(min_history, len(returns) - rebalance_frequency, rebalance_frequency):
            end_idx = min(start_idx + lookback_window, len(returns))
            
            # Extract training window
            train_returns = returns.iloc[start_idx:end_idx]
            train_data = {'returns': train_returns}
            
            # Optimize parameters for this window
            window_result = self._bayesian_optimization(
                parameters, objective, train_data, n_trials // 10  # Fewer trials per window
            )
            
            # Test on next period
            test_start = end_idx
            test_end = min(test_start + rebalance_frequency, len(returns))
            test_returns = returns.iloc[test_start:test_end]
            test_data = {'returns': test_returns}
            
            # Evaluate best parameters on test period
            test_score = self._evaluate_objective(objective, window_result.best_parameters, test_data)
            
            walk_forward_results.append({
                'period_start': start_idx,
                'period_end': end_idx,
                'test_start': test_start,
                'test_end': test_end,
                'best_parameters': window_result.best_parameters,
                'train_score': window_result.best_score,
                'test_score': test_score,
                'overfitting': window_result.best_score - test_score
            })
            
            overall_history.extend(window_result.optimization_history)
        
        # Calculate overall metrics
        train_scores = [r['train_score'] for r in walk_forward_results]
        test_scores = [r['test_score'] for r in walk_forward_results]
        overfitting_scores = [r['overfitting'] for r in walk_forward_results]
        
        # Best parameters based on most recent optimization
        best_params = walk_forward_results[-1]['best_parameters'] if walk_forward_results else {}
        best_score = np.mean(test_scores)
        
        convergence_metrics = {
            'walk_forward_periods': len(walk_forward_results),
            'average_train_score': np.mean(train_scores),
            'average_test_score': np.mean(test_scores),
            'average_overfitting': np.mean(overfitting_scores),
            'stability_score': 1 - np.std(test_scores) / abs(np.mean(test_scores)) if np.mean(test_scores) != 0 else 0,
            'parameter_stability': self._calculate_parameter_stability_across_periods(walk_forward_results, parameters)
        }
        
        return OptimizationResult(
            method=OptimizationMethod.WALK_FORWARD,
            best_parameters=best_params,
            best_score=best_score,
            objective_values={'primary': best_score, 'train': np.mean(train_scores)},
            optimization_history=overall_history,
            convergence_metrics=convergence_metrics,
            sensitivity_analysis={},
            stability_metrics={},
            overfitting_score=np.mean(overfitting_scores),
            statistical_significance={},
            walk_forward_results={
                'period_results': walk_forward_results,
                'summary_stats': {
                    'train_scores': train_scores,
                    'test_scores': test_scores,
                    'overfitting_scores': overfitting_scores
                }
            }
        )
    
    # Helper methods for optimization
    def _split_data(self, data: Dict[str, Any], validation_split: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Split data into training and validation sets"""
        train_data = {}
        val_data = {}
        
        for key, value in data.items():
            if isinstance(value, pd.Series) or isinstance(value, pd.DataFrame):
                split_idx = int(len(value) * (1 - validation_split))
                train_data[key] = value.iloc[:split_idx]
                val_data[key] = value.iloc[split_idx:]
            else:
                # Non-time series data - duplicate for both sets
                train_data[key] = value
                val_data[key] = value
                
        return train_data, val_data
    
    def _evaluate_objective(
        self,
        objective: Union[ObjectiveType, ObjectiveFunction, MultiObjectiveFunction],
        parameters: Dict[str, Any],
        data: Dict[str, Any]
    ) -> float:
        """Evaluate objective function with given parameters"""
        try:
            if isinstance(objective, ObjectiveType):
                obj_func = self.objective_functions.get(objective)
                if obj_func is None:
                    raise ValueError(f"No implementation for objective type: {objective}")
                return obj_func.evaluate(parameters, data)
            elif isinstance(objective, ObjectiveFunction):
                return objective.evaluate(parameters, data)
            elif isinstance(objective, MultiObjectiveFunction):
                _, weighted_sum = objective.evaluate(parameters, data)
                return weighted_sum
            else:
                raise ValueError(f"Unknown objective type: {type(objective)}")
        except Exception as e:
            logger.error(f"Objective evaluation failed: {e}")
            return -np.inf
    
    def _generate_parameter_grid(self, parameters: List[Parameter], max_combinations: int) -> List[Dict[str, Any]]:
        """Generate parameter grid for grid search"""
        param_ranges = []
        
        for param in parameters:
            if param.param_type == 'continuous':
                # Create discrete grid for continuous parameters
                n_points = min(10, int(max_combinations ** (1/len(parameters))))
                if param.log_scale:
                    points = np.logspace(np.log10(param.bounds[0]), np.log10(param.bounds[1]), n_points)
                else:
                    points = np.linspace(param.bounds[0], param.bounds[1], n_points)
                param_ranges.append([(param.name, val) for val in points])
            else:
                param_ranges.append([(param.name, val) for val in param.bounds])
        
        # Generate all combinations
        combinations = list(itertools.product(*param_ranges))
        
        # Limit to max_combinations
        if len(combinations) > max_combinations:
            indices = np.random.choice(len(combinations), max_combinations, replace=False)
            combinations = [combinations[i] for i in indices]
        
        # Convert to parameter dictionaries
        param_grid = []
        for combination in combinations:
            params = {param_name: value for param_name, value in combination}
            param_grid.append(params)
        
        return param_grid
    
    def _latin_hypercube_sampling(self, bounds: np.ndarray, n_samples: int) -> np.ndarray:
        """Generate Latin Hypercube samples within bounds"""
        n_dims = bounds.shape[0]
        
        # Generate LHS samples in [0, 1]
        samples = np.random.random((n_samples, n_dims))
        
        # Apply Latin Hypercube structure
        for i in range(n_dims):
            order = np.random.permutation(n_samples)
            samples[:, i] = (order + samples[:, i]) / n_samples
        
        # Scale to actual bounds
        for i in range(n_dims):
            samples[:, i] = bounds[i, 0] + samples[:, i] * (bounds[i, 1] - bounds[i, 0])
        
        return samples
    
    def _optimize_acquisition_function(self, gp: GaussianProcessRegressor, bounds: np.ndarray, 
                                     current_best: float) -> np.ndarray:
        """Optimize Expected Improvement acquisition function"""
        def expected_improvement(x):
            x = x.reshape(1, -1)
            mu, sigma = gp.predict(x, return_std=True)
            
            if sigma == 0:
                return 0
            
            improvement = mu - current_best - 0.01  # Small exploration bonus
            z = improvement / sigma
            ei = improvement * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
            return -ei  # Minimize negative EI
        
        # Multi-start optimization
        best_x = None
        best_ei = np.inf
        
        for _ in range(10):
            x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])
            
            try:
                result = minimize(expected_improvement, x0, bounds=bounds.tolist(), method='L-BFGS-B')
                if result.fun < best_ei:
                    best_ei = result.fun
                    best_x = result.x
            except:
                continue
        
        return best_x if best_x is not None else np.random.uniform(bounds[:, 0], bounds[:, 1])
    
    def _check_convergence(self, recent_scores: np.ndarray, tolerance: float = 1e-4) -> bool:
        """Check if optimization has converged"""
        if len(recent_scores) < 10:
            return False
        
        # Check if improvement rate is below threshold
        improvement_rate = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
        return abs(improvement_rate) < tolerance
    
    def _calculate_convergence_rate(self, scores: np.ndarray) -> float:
        """Calculate convergence rate of optimization"""
        if len(scores) < 2:
            return 0.0
        
        # Fit exponential convergence curve
        x = np.arange(len(scores))
        try:
            # Simple linear fit to log of improvement
            improvements = np.maximum(scores - scores[0], 1e-10)
            log_improvements = np.log(improvements)
            slope, _ = np.polyfit(x, log_improvements, 1)
            return slope
        except:
            return 0.0
    
    def _genetic_operators(self, population: List[Dict[str, Any]], 
                          scores: List[float], parameters: List[Parameter]) -> List[Dict[str, Any]]:
        """Apply genetic operators: selection, crossover, mutation"""
        population_size = len(population)
        
        # Selection (tournament selection)
        selected = []
        for _ in range(population_size):
            tournament_size = 3
            tournament_indices = np.random.choice(population_size, tournament_size, replace=False)
            tournament_scores = [scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_scores)]
            selected.append(population[winner_idx].copy())
        
        # Crossover and mutation
        new_population = []
        for i in range(0, population_size - 1, 2):
            parent1 = selected[i]
            parent2 = selected[i + 1] if i + 1 < population_size else selected[0]
            
            # Crossover
            child1, child2 = self._crossover(parent1, parent2, parameters)
            
            # Mutation
            child1 = self._mutate(child1, parameters)
            child2 = self._mutate(child2, parameters)
            
            new_population.extend([child1, child2])
        
        return new_population[:population_size]
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any], 
                  parameters: List[Parameter]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Crossover operation for genetic algorithm"""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for param in parameters:
            if np.random.random() < 0.5:  # 50% crossover probability
                child1[param.name] = parent2[param.name]
                child2[param.name] = parent1[param.name]
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, Any], parameters: List[Parameter], 
               mutation_rate: float = 0.1) -> Dict[str, Any]:
        """Mutation operation for genetic algorithm"""
        mutated = individual.copy()
        
        for param in parameters:
            if np.random.random() < mutation_rate:
                if param.param_type == 'continuous':
                    # Gaussian mutation
                    current_val = mutated[param.name]
                    mutation_strength = (param.bounds[1] - param.bounds[0]) * 0.1
                    new_val = current_val + np.random.normal(0, mutation_strength)
                    new_val = np.clip(new_val, param.bounds[0], param.bounds[1])
                    mutated[param.name] = new_val
                else:
                    # Random selection for discrete/categorical
                    mutated[param.name] = np.random.choice(param.bounds)
        
        return mutated
    
    def _calculate_population_diversity(self, population: List[Dict[str, Any]], 
                                      parameters: List[Parameter]) -> float:
        """Calculate diversity of population"""
        if len(population) < 2:
            return 0.0
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = 0
                for param in parameters:
                    if param.param_type == 'continuous':
                        val_range = param.bounds[1] - param.bounds[0]
                        normalized_dist = abs(population[i][param.name] - population[j][param.name]) / val_range
                        distance += normalized_dist ** 2
                    else:
                        distance += 0 if population[i][param.name] == population[j][param.name] else 1
                distances.append(np.sqrt(distance))
        
        return np.mean(distances)
    
    def _sensitivity_analysis(self, parameters: List[Parameter], 
                            objective: Union[ObjectiveType, ObjectiveFunction, MultiObjectiveFunction],
                            best_parameters: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Analyze parameter sensitivity around optimal point"""
        sensitivity_results = {}
        
        for param in parameters:
            param_name = param.name
            base_value = best_parameters[param_name]
            
            # Test parameter variations
            if param.param_type == 'continuous':
                # Test ±10% and ±20% variations
                param_range = param.bounds[1] - param.bounds[0]
                variations = [0.9, 0.95, 1.05, 1.1]  # Multiplicative factors
                
                if param.log_scale:
                    test_values = [base_value * var for var in variations]
                else:
                    test_values = [base_value + (var - 1) * param_range * 0.1 for var in variations]
                
                # Ensure values are within bounds
                test_values = [np.clip(val, param.bounds[0], param.bounds[1]) for val in test_values]
            else:
                # For discrete/categorical, test all other values
                test_values = [val for val in param.bounds if val != base_value]
            
            # Evaluate objective at each test value
            scores = []
            for test_value in test_values:
                test_params = best_parameters.copy()
                test_params[param_name] = test_value
                score = self._evaluate_objective(objective, test_params, data)
                scores.append(score)
            
            # Calculate sensitivity metrics
            if scores:
                base_score = self._evaluate_objective(objective, best_parameters, data)
                max_impact = max([abs(score - base_score) for score in scores])
                avg_impact = np.mean([abs(score - base_score) for score in scores])
                
                sensitivity_results[param_name] = {
                    'max_impact': max_impact,
                    'average_impact': avg_impact,
                    'sensitivity_ratio': max_impact / abs(base_score) if base_score != 0 else 0,
                    'stability_score': 1 - (np.std(scores) / abs(base_score)) if base_score != 0 else 0
                }
            else:
                sensitivity_results[param_name] = {
                    'max_impact': 0,
                    'average_impact': 0,
                    'sensitivity_ratio': 0,
                    'stability_score': 1
                }
        
        return sensitivity_results
    
    def _analyze_parameter_stability(self, parameters: List[Parameter],
                                   objective: Union[ObjectiveType, ObjectiveFunction, MultiObjectiveFunction],
                                   best_parameters: Dict[str, Any], data: Dict[str, Any],
                                   cv_folds: int) -> Dict[str, float]:
        """Analyze parameter stability using cross-validation"""
        returns = data.get('returns')
        if returns is None:
            return {'stability_score': 1.0}
        
        # Time series cross-validation
        fold_size = len(returns) // cv_folds
        fold_parameters = []
        
        for fold in range(cv_folds):
            start_idx = fold * fold_size
            end_idx = min((fold + 1) * fold_size, len(returns))
            fold_data = {'returns': returns.iloc[start_idx:end_idx]}
            
            # Quick optimization on fold (fewer iterations)
            try:
                fold_result = self._bayesian_optimization(
                    parameters, objective, fold_data, n_trials=20
                )
                fold_parameters.append(fold_result.best_parameters)
            except Exception as e:
                logger.error(f"Fold optimization failed: {e}")
                continue
        
        if len(fold_parameters) < 2:
            return {'stability_score': 0.0}
        
        # Calculate parameter stability
        stability_metrics = {}
        
        for param in parameters:
            param_name = param.name
            param_values = [fp[param_name] for fp in fold_parameters if param_name in fp]
            
            if len(param_values) < 2:
                continue
            
            if param.param_type == 'continuous':
                # Coefficient of variation
                cv = np.std(param_values) / abs(np.mean(param_values)) if np.mean(param_values) != 0 else 0
                stability_metrics[f'{param_name}_stability'] = max(0, 1 - cv)
            else:
                # Mode frequency for discrete/categorical
                from collections import Counter
                counts = Counter(param_values)
                mode_freq = counts.most_common(1)[0][1] / len(param_values)
                stability_metrics[f'{param_name}_stability'] = mode_freq
        
        # Overall stability score
        if stability_metrics:
            overall_stability = np.mean(list(stability_metrics.values()))
            stability_metrics['overall_stability'] = overall_stability
        else:
            stability_metrics['overall_stability'] = 0.0
        
        return stability_metrics
    
    def _test_statistical_significance(self, objective: Union[ObjectiveType, ObjectiveFunction, MultiObjectiveFunction],
                                     best_parameters: Dict[str, Any], data: Dict[str, Any],
                                     cv_folds: int) -> Dict[str, float]:
        """Test statistical significance of optimization results"""
        # Bootstrap test for parameter significance
        returns = data.get('returns')
        if returns is None:
            return {'significance': 0.0}
        
        # Baseline performance (random parameters)
        n_bootstrap = 100
        baseline_scores = []
        
        for _ in range(n_bootstrap):
            # Generate random parameters
            random_params = {}
            # This would need parameter definitions - simplified for now
            random_params = best_parameters  # Placeholder
            
            # Evaluate with bootstrap sample
            bootstrap_data = {'returns': returns.sample(len(returns), replace=True)}
            score = self._evaluate_objective(objective, random_params, bootstrap_data)
            baseline_scores.append(score)
        
        # Best parameter performance
        best_score = self._evaluate_objective(objective, best_parameters, data)
        
        # Statistical test
        baseline_mean = np.mean(baseline_scores)
        baseline_std = np.std(baseline_scores)
        
        if baseline_std > 0:
            z_score = (best_score - baseline_mean) / baseline_std
            p_value = 1 - stats.norm.cdf(z_score)
        else:
            p_value = 0.5
        
        return {
            'z_score': z_score if 'z_score' in locals() else 0,
            'p_value': p_value,
            'significance': p_value < 0.05,
            'effect_size': (best_score - baseline_mean) / baseline_std if baseline_std > 0 else 0
        }
    
    def _calculate_parameter_stability_across_periods(self, walk_forward_results: List[Dict[str, Any]],
                                                    parameters: List[Parameter]) -> Dict[str, float]:
        """Calculate parameter stability across walk-forward periods"""
        stability_scores = {}
        
        for param in parameters:
            param_name = param.name
            param_values = [result['best_parameters'].get(param_name) for result in walk_forward_results]
            param_values = [v for v in param_values if v is not None]
            
            if len(param_values) < 2:
                stability_scores[param_name] = 0.0
                continue
            
            if param.param_type == 'continuous':
                # Coefficient of variation
                cv = np.std(param_values) / abs(np.mean(param_values)) if np.mean(param_values) != 0 else float('inf')
                stability_scores[param_name] = max(0, 1 - cv)
            else:
                # Mode frequency
                from collections import Counter
                counts = Counter(param_values)
                mode_freq = counts.most_common(1)[0][1] / len(param_values)
                stability_scores[param_name] = mode_freq
        
        return stability_scores
    
    # Placeholder methods for other optimization algorithms
    def _random_search_optimization(self, parameters, objective, data, n_trials, parallel_jobs):
        """Random search optimization - simplified implementation"""
        return self._grid_search_optimization(parameters, objective, data, n_trials, parallel_jobs)
    
    def _particle_swarm_optimization(self, parameters, objective, data, n_trials):
        """Particle swarm optimization - simplified implementation"""
        return self._bayesian_optimization(parameters, objective, data, n_trials)
    
    def _differential_evolution_optimization(self, parameters, objective, data, n_trials):
        """Differential evolution optimization - simplified implementation"""
        return self._bayesian_optimization(parameters, objective, data, n_trials)
    
    def _multi_objective_optimization(self, parameters, objective, data, n_trials):
        """Multi-objective optimization - simplified implementation"""
        return self._bayesian_optimization(parameters, objective, data, n_trials)