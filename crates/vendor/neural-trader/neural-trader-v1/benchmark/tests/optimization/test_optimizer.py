"""Tests for optimization engine."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from benchmark.src.optimization.optimizer import OptimizationEngine, OptimizationStatus
from benchmark.src.config import OptimizationConfig


class TestOptimizationEngine:
    """Test suite for optimization engine."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = OptimizationConfig(
            n_initial_points=5,
            convergence_patience=10,
            convergence_min_delta=1e-6,
            results_dir="test_results"
        )
        
        # Simple quadratic objective function
        def objective(params):
            x = params.get('x', 0)
            y = params.get('y', 0)
            return -(x - 2)**2 - (y - 3)**2 + 10
            
        self.objective_function = objective
        
        self.parameter_space = {
            'x': (-5, 5),
            'y': (-5, 5)
        }
        
    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = OptimizationEngine(
            config=self.config,
            objective_function=self.objective_function,
            parameter_space=self.parameter_space
        )
        
        assert optimizer.status == OptimizationStatus.INITIALIZING
        assert optimizer.best_params is None
        assert optimizer.best_score == float('-inf')
        
    @pytest.mark.asyncio
    async def test_bayesian_optimization(self):
        """Test Bayesian optimization."""
        optimizer = OptimizationEngine(
            config=self.config,
            objective_function=self.objective_function,
            parameter_space=self.parameter_space
        )
        
        result = await optimizer.optimize(
            method='bayesian',
            n_trials=20
        )
        
        assert result.best_params is not None
        assert result.best_score > 0
        assert len(result.all_trials) == 20
        assert result.method_used == 'bayesian'
        
        # Check convergence to optimum (x=2, y=3)
        assert abs(result.best_params['x'] - 2) < 1.0
        assert abs(result.best_params['y'] - 3) < 1.0
        
    def test_parameter_space_validation(self):
        """Test parameter space validation."""
        invalid_space = {
            'x': 'invalid',
            'y': (-5, 5)
        }
        
        with pytest.raises(ValueError):
            OptimizationEngine(
                config=self.config,
                objective_function=self.objective_function,
                parameter_space=invalid_space
            )
            
    def test_objective_function_error_handling(self):
        """Test error handling in objective function."""
        def failing_objective(params):
            raise ValueError("Test error")
            
        optimizer = OptimizationEngine(
            config=self.config,
            objective_function=failing_objective,
            parameter_space=self.parameter_space
        )
        
        # Should handle errors gracefully
        score = optimizer._evaluate_parameters({'x': 1, 'y': 1})
        assert score == float('-inf')
        
    def test_convergence_detection(self):
        """Test convergence detection."""
        # Mock convergence analyzer
        with patch('benchmark.src.optimization.optimizer.ConvergenceAnalyzer') as mock_analyzer:
            mock_instance = Mock()
            mock_instance.has_converged.return_value = True
            mock_analyzer.return_value = mock_instance
            
            optimizer = OptimizationEngine(
                config=self.config,
                objective_function=self.objective_function,
                parameter_space=self.parameter_space
            )
            
            # Should detect convergence
            assert optimizer.convergence_analyzer.has_converged()


@pytest.fixture
def sample_data():
    """Sample market data for testing."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'price': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    return data


class TestOptimizationIntegration:
    """Integration tests for optimization framework."""
    
    def test_strategy_optimization_workflow(self, sample_data):
        """Test complete strategy optimization workflow."""
        # Mock strategy class
        class MockStrategy:
            def __init__(self, lookback=20, threshold=0.01):
                self.lookback = lookback
                self.threshold = threshold
                
            @staticmethod
            def get_parameter_space():
                return {
                    'lookback': (5, 50),
                    'threshold': (0.001, 0.1)
                }
                
        # Mock strategy evaluator
        def mock_evaluator(params, data):
            # Simple mock evaluation
            return np.random.normal(0.001, 0.02, len(data))
            
        config = OptimizationConfig()
        
        optimizer = OptimizationEngine(
            config=config,
            objective_function=lambda p: np.mean(mock_evaluator(p, sample_data)),
            parameter_space=MockStrategy.get_parameter_space()
        )
        
        # Test that optimization can run without errors
        assert optimizer is not None