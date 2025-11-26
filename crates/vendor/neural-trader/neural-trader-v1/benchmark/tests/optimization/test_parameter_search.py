"""
Test suite for parameter optimization algorithms.

Tests grid search, random search, and smart search strategies
for optimizing trading system parameters.
"""

import pytest
import numpy as np
from typing import Dict, List, Tuple, Any
from unittest.mock import Mock, patch

from benchmark.src.optimization.parameter_optimizer import (
    ParameterOptimizer,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    SmartSearchOptimizer,
    OptimizationResult
)


class TestParameterOptimizer:
    """Test base parameter optimizer functionality."""
    
    @pytest.fixture
    def objective_function(self):
        """Simple quadratic objective function for testing."""
        def f(params: Dict[str, float]) -> float:
            # Minimize (x-2)^2 + (y-3)^2
            return (params['x'] - 2) ** 2 + (params['y'] - 3) ** 2
        return f
    
    @pytest.fixture
    def search_space(self):
        """Define parameter search space."""
        return {
            'x': {'type': 'float', 'min': -10, 'max': 10},
            'y': {'type': 'float', 'min': -10, 'max': 10}
        }
    
    @pytest.fixture
    def trading_objective(self):
        """Realistic trading strategy objective function."""
        def f(params: Dict[str, float]) -> float:
            # Simulate Sharpe ratio calculation
            ma_short = params['ma_short']
            ma_long = params['ma_long']
            stop_loss = params['stop_loss']
            
            # Penalize invalid parameter combinations
            if ma_short >= ma_long:
                return -10.0
            
            # Simulate performance based on parameters
            sharpe = 1.5 - 0.01 * abs(ma_short - 20) - 0.005 * abs(ma_long - 50)
            sharpe -= 0.1 * abs(stop_loss - 0.02)
            
            return sharpe
        return f
    
    @pytest.fixture
    def trading_search_space(self):
        """Trading strategy parameter space."""
        return {
            'ma_short': {'type': 'int', 'min': 5, 'max': 50},
            'ma_long': {'type': 'int', 'min': 20, 'max': 200},
            'stop_loss': {'type': 'float', 'min': 0.001, 'max': 0.1},
            'take_profit': {'type': 'float', 'min': 0.01, 'max': 0.5}
        }


class TestGridSearchOptimizer(TestParameterOptimizer):
    """Test grid search optimization."""
    
    def test_grid_search_initialization(self, search_space):
        """Test grid search optimizer initialization."""
        optimizer = GridSearchOptimizer(
            search_space=search_space,
            resolution={'x': 5, 'y': 5}
        )
        
        assert optimizer.search_space == search_space
        assert optimizer.resolution == {'x': 5, 'y': 5}
        assert optimizer.total_evaluations == 25
    
    def test_grid_search_optimization(self, objective_function, search_space):
        """Test grid search finds optimal parameters."""
        optimizer = GridSearchOptimizer(
            search_space=search_space,
            resolution={'x': 20, 'y': 20}
        )
        
        result = optimizer.optimize(
            objective_function=objective_function,
            max_evaluations=400
        )
        
        assert isinstance(result, OptimizationResult)
        assert abs(result.best_params['x'] - 2.0) < 1.0
        assert abs(result.best_params['y'] - 3.0) < 1.0
        assert result.best_score < 2.0
        assert len(result.history) == 400
    
    def test_grid_search_with_constraints(self, trading_objective, trading_search_space):
        """Test grid search with parameter constraints."""
        def constraint(params):
            # Ensure ma_long > ma_short
            return params['ma_long'] > params['ma_short']
        
        optimizer = GridSearchOptimizer(
            search_space=trading_search_space,
            resolution={'ma_short': 10, 'ma_long': 10, 'stop_loss': 5, 'take_profit': 5},
            constraints=[constraint]
        )
        
        result = optimizer.optimize(
            objective_function=trading_objective,
            max_evaluations=1000
        )
        
        assert result.best_params['ma_long'] > result.best_params['ma_short']
        assert result.best_score > 0
    
    def test_grid_search_early_stopping(self, objective_function, search_space):
        """Test early stopping when target is reached."""
        optimizer = GridSearchOptimizer(
            search_space=search_space,
            resolution={'x': 50, 'y': 50}
        )
        
        result = optimizer.optimize(
            objective_function=objective_function,
            max_evaluations=2500,
            target_score=0.1
        )
        
        assert result.best_score < 0.1
        assert len(result.history) < 2500
        assert result.converged


class TestRandomSearchOptimizer(TestParameterOptimizer):
    """Test random search optimization."""
    
    def test_random_search_initialization(self, search_space):
        """Test random search optimizer initialization."""
        optimizer = RandomSearchOptimizer(
            search_space=search_space,
            seed=42
        )
        
        assert optimizer.search_space == search_space
        assert optimizer.rng.integers(0, 100) == 51  # Verify seed is set
    
    def test_random_search_optimization(self, objective_function, search_space):
        """Test random search finds good parameters."""
        optimizer = RandomSearchOptimizer(
            search_space=search_space,
            seed=42
        )
        
        result = optimizer.optimize(
            objective_function=objective_function,
            max_evaluations=100
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.best_score < 5.0  # Should find reasonable solution
        assert len(result.history) == 100
    
    def test_random_search_convergence(self, objective_function, search_space):
        """Test random search convergence with many evaluations."""
        optimizer = RandomSearchOptimizer(
            search_space=search_space,
            seed=123
        )
        
        result = optimizer.optimize(
            objective_function=objective_function,
            max_evaluations=1000
        )
        
        assert abs(result.best_params['x'] - 2.0) < 0.5
        assert abs(result.best_params['y'] - 3.0) < 0.5
        assert result.best_score < 0.5
    
    def test_random_search_with_categorical(self):
        """Test random search with categorical parameters."""
        search_space = {
            'strategy': {'type': 'categorical', 'values': ['momentum', 'mean_reversion', 'breakout']},
            'timeframe': {'type': 'categorical', 'values': ['1m', '5m', '15m', '1h']},
            'threshold': {'type': 'float', 'min': 0.0, 'max': 1.0}
        }
        
        def objective(params):
            score = 0.0
            if params['strategy'] == 'momentum':
                score += 0.5
            if params['timeframe'] == '5m':
                score += 0.3
            score += params['threshold'] * 0.2
            return score
        
        optimizer = RandomSearchOptimizer(search_space=search_space)
        result = optimizer.optimize(objective, max_evaluations=100)
        
        assert result.best_params['strategy'] in ['momentum', 'mean_reversion', 'breakout']
        assert result.best_params['timeframe'] in ['1m', '5m', '15m', '1h']
        assert 0 <= result.best_params['threshold'] <= 1


class TestSmartSearchOptimizer(TestParameterOptimizer):
    """Test smart search optimization (adaptive strategies)."""
    
    def test_smart_search_initialization(self, search_space):
        """Test smart search optimizer initialization."""
        optimizer = SmartSearchOptimizer(
            search_space=search_space,
            initial_samples=10,
            exploitation_rate=0.7
        )
        
        assert optimizer.search_space == search_space
        assert optimizer.initial_samples == 10
        assert optimizer.exploitation_rate == 0.7
    
    def test_smart_search_adaptive_behavior(self, objective_function, search_space):
        """Test smart search adapts based on results."""
        optimizer = SmartSearchOptimizer(
            search_space=search_space,
            initial_samples=20,
            exploitation_rate=0.8
        )
        
        result = optimizer.optimize(
            objective_function=objective_function,
            max_evaluations=100
        )
        
        # Smart search should outperform random search
        assert result.best_score < 1.0
        assert len(result.history) == 100
        
        # Check that later samples are concentrated near optimum
        later_samples = result.history[50:]
        x_values = [h['params']['x'] for h in later_samples]
        y_values = [h['params']['y'] for h in later_samples]
        
        assert np.std(x_values) < 2.0  # Should focus search
        assert np.std(y_values) < 2.0
    
    def test_smart_search_performance_tracking(self, trading_objective, trading_search_space):
        """Test smart search tracks performance metrics."""
        optimizer = SmartSearchOptimizer(
            search_space=trading_search_space,
            track_metrics=True
        )
        
        result = optimizer.optimize(
            objective_function=trading_objective,
            max_evaluations=200
        )
        
        assert 'convergence_rate' in result.metrics
        assert 'exploration_efficiency' in result.metrics
        assert 'best_score_progression' in result.metrics
        assert len(result.metrics['best_score_progression']) > 0


class TestOptimizationResult:
    """Test optimization result handling."""
    
    def test_optimization_result_creation(self):
        """Test creating optimization result."""
        result = OptimizationResult(
            best_params={'x': 2.0, 'y': 3.0},
            best_score=0.1,
            history=[],
            converged=True,
            iterations=100,
            metrics={'time_elapsed': 5.2}
        )
        
        assert result.best_params == {'x': 2.0, 'y': 3.0}
        assert result.best_score == 0.1
        assert result.converged
        assert result.iterations == 100
        assert result.metrics['time_elapsed'] == 5.2
    
    def test_optimization_result_export(self):
        """Test exporting optimization results."""
        history = [
            {'params': {'x': 1.0}, 'score': 0.5, 'iteration': 1},
            {'params': {'x': 2.0}, 'score': 0.1, 'iteration': 2}
        ]
        
        result = OptimizationResult(
            best_params={'x': 2.0},
            best_score=0.1,
            history=history,
            converged=True,
            iterations=2
        )
        
        export = result.to_dict()
        assert export['best_params'] == {'x': 2.0}
        assert export['best_score'] == 0.1
        assert len(export['history']) == 2
        assert export['converged'] is True
    
    def test_optimization_result_summary(self):
        """Test generating optimization summary."""
        result = OptimizationResult(
            best_params={'ma_short': 20, 'ma_long': 50, 'stop_loss': 0.02},
            best_score=1.45,
            history=[{'score': 1.0}, {'score': 1.2}, {'score': 1.45}],
            converged=True,
            iterations=3,
            metrics={'time_elapsed': 10.5, 'evaluations_per_second': 0.28}
        )
        
        summary = result.summary()
        assert 'Best Parameters' in summary
        assert 'ma_short: 20' in summary
        assert 'Best Score: 1.45' in summary
        assert 'Converged: True' in summary
        assert 'Time Elapsed: 10.5' in summary


class TestParameterValidation:
    """Test parameter validation and constraints."""
    
    def test_parameter_bounds_validation(self):
        """Test parameter bounds are enforced."""
        search_space = {
            'x': {'type': 'float', 'min': 0, 'max': 1},
            'y': {'type': 'int', 'min': 10, 'max': 100}
        }
        
        optimizer = GridSearchOptimizer(search_space=search_space)
        
        # Test valid parameters
        assert optimizer.validate_params({'x': 0.5, 'y': 50})
        
        # Test invalid parameters
        assert not optimizer.validate_params({'x': 1.5, 'y': 50})
        assert not optimizer.validate_params({'x': 0.5, 'y': 5})
        assert not optimizer.validate_params({'x': 0.5})  # Missing param
    
    def test_custom_constraints(self):
        """Test custom parameter constraints."""
        search_space = {
            'a': {'type': 'float', 'min': 0, 'max': 10},
            'b': {'type': 'float', 'min': 0, 'max': 10}
        }
        
        # Constraint: a + b <= 10
        constraint = lambda p: p['a'] + p['b'] <= 10
        
        optimizer = RandomSearchOptimizer(
            search_space=search_space,
            constraints=[constraint]
        )
        
        # Generate many samples and check constraint
        valid_samples = []
        for _ in range(100):
            params = optimizer.sample_params()
            if optimizer.validate_params(params):
                valid_samples.append(params)
        
        # All valid samples should satisfy constraint
        for params in valid_samples:
            assert params['a'] + params['b'] <= 10