"""
Test suite for optimization convergence behavior.

Tests convergence criteria, early stopping, and convergence analysis
for various optimization algorithms.
"""

import pytest
import numpy as np
from typing import List, Dict, Callable
from unittest.mock import Mock, patch

from benchmark.src.optimization.genetic_optimizer import GeneticOptimizer
from benchmark.src.optimization.bayesian_optimizer import BayesianOptimizer
from benchmark.src.optimization.ml_optimizer import MLOptimizer
from benchmark.src.optimization.parameter_optimizer import OptimizationResult


class TestConvergenceCriteria:
    """Test various convergence criteria."""
    
    @pytest.fixture
    def quadratic_objective(self):
        """Simple quadratic function with known minimum."""
        def f(params: Dict[str, float]) -> float:
            return sum((params[k] - 2.0) ** 2 for k in params)
        return f
    
    @pytest.fixture
    def noisy_objective(self):
        """Objective function with noise."""
        np.random.seed(42)
        def f(params: Dict[str, float]) -> float:
            base = sum((params[k] - 2.0) ** 2 for k in params)
            noise = np.random.normal(0, 0.1)
            return base + noise
        return f
    
    @pytest.fixture
    def search_space(self):
        """Standard search space for testing."""
        return {
            'x': {'type': 'float', 'min': -10, 'max': 10},
            'y': {'type': 'float', 'min': -10, 'max': 10},
            'z': {'type': 'float', 'min': -10, 'max': 10}
        }
    
    def test_absolute_tolerance_convergence(self, quadratic_objective, search_space):
        """Test convergence based on absolute tolerance."""
        optimizer = GeneticOptimizer(
            search_space=search_space,
            population_size=50
        )
        
        result = optimizer.optimize(
            objective_function=quadratic_objective,
            max_evaluations=1000,
            convergence_tolerance=0.01,
            convergence_window=10
        )
        
        assert result.converged
        assert result.best_score < 0.01
        assert result.convergence_iteration < 1000
    
    def test_relative_tolerance_convergence(self, quadratic_objective, search_space):
        """Test convergence based on relative improvement."""
        optimizer = BayesianOptimizer(
            search_space=search_space,
            initial_samples=20
        )
        
        result = optimizer.optimize(
            objective_function=quadratic_objective,
            max_evaluations=500,
            relative_tolerance=0.001,  # 0.1% improvement
            convergence_window=20
        )
        
        # Check relative improvement in last window
        scores = [h['score'] for h in result.history[-20:]]
        relative_change = abs(scores[-1] - scores[0]) / abs(scores[0])
        assert relative_change < 0.001
    
    def test_patience_based_convergence(self, noisy_objective, search_space):
        """Test convergence with patience for noisy objectives."""
        optimizer = MLOptimizer(
            search_space=search_space,
            model_type='random_forest'
        )
        
        result = optimizer.optimize(
            objective_function=noisy_objective,
            max_evaluations=500,
            patience=50,  # Stop after 50 iterations without improvement
            min_improvement=0.01
        )
        
        # Find best score iteration
        best_iteration = 0
        best_score = float('inf')
        for i, h in enumerate(result.history):
            if h['score'] < best_score:
                best_score = h['score']
                best_iteration = i
        
        # Should stop within patience window after best score
        assert len(result.history) <= best_iteration + 50 + 10  # Some buffer
    
    def test_multi_criteria_convergence(self, quadratic_objective, search_space):
        """Test convergence with multiple criteria."""
        optimizer = GeneticOptimizer(
            search_space=search_space,
            population_size=30
        )
        
        result = optimizer.optimize(
            objective_function=quadratic_objective,
            max_evaluations=1000,
            convergence_criteria={
                'absolute_tolerance': 0.1,
                'relative_tolerance': 0.01,
                'min_iterations': 100,
                'patience': 30
            }
        )
        
        assert result.converged
        assert result.iterations >= 100  # Minimum iterations
        assert result.best_score < 0.1  # Absolute tolerance
    
    def test_population_diversity_convergence(self, quadratic_objective, search_space):
        """Test convergence based on population diversity (genetic algorithms)."""
        optimizer = GeneticOptimizer(
            search_space=search_space,
            population_size=50,
            track_diversity=True
        )
        
        result = optimizer.optimize(
            objective_function=quadratic_objective,
            max_evaluations=1000,
            diversity_threshold=0.01
        )
        
        assert result.converged
        assert 'population_diversity' in result.metrics
        assert result.metrics['population_diversity'][-1] < 0.01


class TestEarlyStopping:
    """Test early stopping mechanisms."""
    
    def test_target_score_early_stopping(self):
        """Test stopping when target score is reached."""
        search_space = {'x': {'type': 'float', 'min': -5, 'max': 5}}
        
        def objective(params):
            return (params['x'] - 2) ** 2
        
        optimizer = BayesianOptimizer(search_space=search_space)
        result = optimizer.optimize(
            objective_function=objective,
            max_evaluations=1000,
            target_score=0.01
        )
        
        assert result.best_score <= 0.01
        assert result.early_stopped
        assert len(result.history) < 1000
    
    def test_time_based_early_stopping(self):
        """Test stopping based on time limit."""
        search_space = {
            'x': {'type': 'float', 'min': -10, 'max': 10},
            'y': {'type': 'float', 'min': -10, 'max': 10}
        }
        
        def slow_objective(params):
            import time
            time.sleep(0.01)  # Simulate slow evaluation
            return params['x'] ** 2 + params['y'] ** 2
        
        optimizer = MLOptimizer(search_space=search_space)
        result = optimizer.optimize(
            objective_function=slow_objective,
            max_evaluations=1000,
            time_limit=1.0  # 1 second limit
        )
        
        assert result.time_stopped
        assert result.metrics['time_elapsed'] <= 1.5  # Some buffer
    
    def test_callback_based_early_stopping(self):
        """Test custom callback for early stopping."""
        search_space = {'x': {'type': 'float', 'min': 0, 'max': 10}}
        
        def objective(params):
            return -params['x']  # Maximize x
        
        stop_requested = False
        
        def should_stop(iteration, best_score, history):
            # Stop if we find x > 8
            nonlocal stop_requested
            if best_score < -8:
                stop_requested = True
                return True
            return False
        
        optimizer = GeneticOptimizer(search_space=search_space)
        result = optimizer.optimize(
            objective_function=objective,
            max_evaluations=1000,
            early_stop_callback=should_stop
        )
        
        assert stop_requested
        assert result.best_params['x'] > 8
        assert result.callback_stopped


class TestConvergenceAnalysis:
    """Test convergence analysis and diagnostics."""
    
    def test_convergence_rate_analysis(self):
        """Test analyzing convergence rate."""
        # Create synthetic optimization history
        iterations = 100
        history = []
        
        # Exponential convergence
        for i in range(iterations):
            score = 10 * np.exp(-0.05 * i) + 0.1
            history.append({
                'iteration': i,
                'score': score,
                'params': {'x': 2 - score}
            })
        
        result = OptimizationResult(
            best_params={'x': 2.0},
            best_score=0.1,
            history=history,
            converged=True,
            iterations=iterations
        )
        
        # Analyze convergence
        from benchmark.src.optimization.convergence_analyzer import ConvergenceAnalyzer
        analyzer = ConvergenceAnalyzer()
        
        analysis = analyzer.analyze(result)
        
        assert 'convergence_rate' in analysis
        assert 'convergence_type' in analysis
        assert analysis['convergence_type'] == 'exponential'
        assert analysis['convergence_rate'] > 0
    
    def test_plateau_detection(self):
        """Test detection of optimization plateaus."""
        # Create history with plateau
        history = []
        
        # Initial descent
        for i in range(50):
            score = 10 - 0.2 * i
            history.append({'iteration': i, 'score': score})
        
        # Plateau
        for i in range(50, 100):
            score = 0.5 + np.random.normal(0, 0.01)
            history.append({'iteration': i, 'score': score})
        
        result = OptimizationResult(
            best_params={'x': 1.0},
            best_score=0.5,
            history=history,
            converged=False,
            iterations=100
        )
        
        from benchmark.src.optimization.convergence_analyzer import ConvergenceAnalyzer
        analyzer = ConvergenceAnalyzer()
        
        plateaus = analyzer.detect_plateaus(result, window_size=10, threshold=0.01)
        
        assert len(plateaus) > 0
        assert plateaus[0]['start'] >= 45
        assert plateaus[0]['duration'] >= 30
    
    def test_convergence_prediction(self):
        """Test predicting convergence based on current progress."""
        # Create partial optimization history
        history = []
        for i in range(50):
            score = 10 * np.exp(-0.1 * i)
            history.append({
                'iteration': i,
                'score': score,
                'params': {'x': score}
            })
        
        partial_result = OptimizationResult(
            best_params={'x': history[-1]['params']['x']},
            best_score=history[-1]['score'],
            history=history,
            converged=False,
            iterations=50
        )
        
        from benchmark.src.optimization.convergence_analyzer import ConvergenceAnalyzer
        analyzer = ConvergenceAnalyzer()
        
        prediction = analyzer.predict_convergence(
            partial_result,
            target_score=0.01,
            confidence_level=0.95
        )
        
        assert 'estimated_iterations' in prediction
        assert 'confidence_interval' in prediction
        assert prediction['estimated_iterations'] > 50
        assert prediction['will_converge'] is True
    
    def test_multi_run_convergence_statistics(self):
        """Test analyzing convergence across multiple optimization runs."""
        # Simulate multiple optimization runs
        runs = []
        
        for seed in range(10):
            np.random.seed(seed)
            history = []
            
            # Each run has slightly different convergence
            rate = 0.05 + np.random.uniform(-0.02, 0.02)
            for i in range(100):
                score = 10 * np.exp(-rate * i) + np.random.normal(0, 0.1)
                history.append({'iteration': i, 'score': score})
            
            result = OptimizationResult(
                best_params={'x': 2.0},
                best_score=history[-1]['score'],
                history=history,
                converged=True,
                iterations=100
            )
            runs.append(result)
        
        from benchmark.src.optimization.convergence_analyzer import ConvergenceAnalyzer
        analyzer = ConvergenceAnalyzer()
        
        stats = analyzer.multi_run_statistics(runs)
        
        assert 'mean_convergence_iteration' in stats
        assert 'std_convergence_iteration' in stats
        assert 'success_rate' in stats
        assert stats['success_rate'] == 1.0  # All runs converged
        assert 60 < stats['mean_convergence_iteration'] < 80


class TestAdaptiveConvergence:
    """Test adaptive convergence strategies."""
    
    def test_adaptive_tolerance(self):
        """Test adapting convergence tolerance based on progress."""
        search_space = {
            'x': {'type': 'float', 'min': -10, 'max': 10},
            'y': {'type': 'float', 'min': -10, 'max': 10}
        }
        
        def objective(params):
            # Function with different scales in different regions
            if abs(params['x']) < 1 and abs(params['y']) < 1:
                return 0.001 * (params['x'] ** 2 + params['y'] ** 2)
            else:
                return params['x'] ** 2 + params['y'] ** 2
        
        optimizer = MLOptimizer(
            search_space=search_space,
            adaptive_convergence=True
        )
        
        result = optimizer.optimize(
            objective_function=objective,
            max_evaluations=500,
            initial_tolerance=1.0,
            final_tolerance=0.0001
        )
        
        assert result.converged
        assert result.best_score < 0.001
        
        # Check that tolerance was adapted
        assert 'tolerance_history' in result.metrics
        assert result.metrics['tolerance_history'][0] == 1.0
        assert result.metrics['tolerance_history'][-1] < 0.01
    
    def test_restart_on_stagnation(self):
        """Test restarting optimization when stagnant."""
        search_space = {'x': {'type': 'float', 'min': -10, 'max': 10}}
        
        # Multi-modal function with local minima
        def objective(params):
            x = params['x']
            return x ** 2 - 10 * np.cos(2 * np.pi * x)
        
        optimizer = GeneticOptimizer(
            search_space=search_space,
            population_size=20,
            enable_restart=True,
            stagnation_threshold=50
        )
        
        result = optimizer.optimize(
            objective_function=objective,
            max_evaluations=1000
        )
        
        assert 'restart_count' in result.metrics
        assert result.metrics['restart_count'] > 0
        assert result.best_params['x'] < 0.1  # Should find global minimum near 0