"""
Test suite for optimization quality metrics.

Tests the quality of optimization results, including solution accuracy,
robustness, and performance across different problem types.
"""

import pytest
import numpy as np
from typing import Dict, List, Callable, Tuple
import time
from unittest.mock import Mock, patch

from benchmark.src.optimization.genetic_optimizer import GeneticOptimizer
from benchmark.src.optimization.bayesian_optimizer import BayesianOptimizer
from benchmark.src.optimization.ml_optimizer import MLOptimizer
from benchmark.src.optimization.parameter_optimizer import (
    GridSearchOptimizer,
    RandomSearchOptimizer,
    SmartSearchOptimizer
)


class TestOptimizationBenchmarks:
    """Benchmark different optimizers on standard test functions."""
    
    @pytest.fixture
    def rosenbrock_function(self):
        """Rosenbrock function - classic optimization benchmark."""
        def f(params: Dict[str, float]) -> float:
            x, y = params['x'], params['y']
            return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
        return f
    
    @pytest.fixture
    def rastrigin_function(self):
        """Rastrigin function - highly multi-modal."""
        def f(params: Dict[str, float]) -> float:
            result = 10 * len(params)
            for key, value in params.items():
                result += value ** 2 - 10 * np.cos(2 * np.pi * value)
            return result
        return f
    
    @pytest.fixture
    def ackley_function(self):
        """Ackley function - many local minima."""
        def f(params: Dict[str, float]) -> float:
            values = list(params.values())
            n = len(values)
            sum_sq = sum(v ** 2 for v in values)
            sum_cos = sum(np.cos(2 * np.pi * v) for v in values)
            
            term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
            term2 = -np.exp(sum_cos / n)
            return term1 + term2 + 20 + np.e
        return f
    
    @pytest.fixture
    def standard_search_space(self):
        """Standard 2D search space."""
        return {
            'x': {'type': 'float', 'min': -5, 'max': 5},
            'y': {'type': 'float', 'min': -5, 'max': 5}
        }
    
    @pytest.fixture
    def high_dim_search_space(self):
        """High-dimensional search space."""
        return {
            f'x{i}': {'type': 'float', 'min': -5, 'max': 5}
            for i in range(10)
        }
    
    def test_optimizer_comparison_rosenbrock(self, rosenbrock_function, standard_search_space):
        """Compare optimizers on Rosenbrock function."""
        optimizers = [
            GridSearchOptimizer(standard_search_space, resolution={'x': 50, 'y': 50}),
            RandomSearchOptimizer(standard_search_space, seed=42),
            SmartSearchOptimizer(standard_search_space, initial_samples=20),
            GeneticOptimizer(standard_search_space, population_size=50),
            BayesianOptimizer(standard_search_space, initial_samples=20),
            MLOptimizer(standard_search_space, model_type='random_forest')
        ]
        
        results = {}
        for optimizer in optimizers:
            start_time = time.time()
            result = optimizer.optimize(
                objective_function=rosenbrock_function,
                max_evaluations=500
            )
            end_time = time.time()
            
            results[optimizer.__class__.__name__] = {
                'best_score': result.best_score,
                'best_params': result.best_params,
                'time': end_time - start_time,
                'evaluations': len(result.history)
            }
        
        # All optimizers should find reasonable solution
        for name, result in results.items():
            assert result['best_score'] < 10.0, f"{name} failed to find good solution"
            
        # Bayesian and ML optimizers should perform best
        assert results['BayesianOptimizer']['best_score'] < 1.0
        assert results['MLOptimizer']['best_score'] < 1.0
    
    def test_scalability_high_dimensions(self, rastrigin_function, high_dim_search_space):
        """Test optimizer performance in high dimensions."""
        # Only test scalable optimizers
        optimizers = [
            RandomSearchOptimizer(high_dim_search_space, seed=42),
            GeneticOptimizer(high_dim_search_space, population_size=100),
            BayesianOptimizer(high_dim_search_space, initial_samples=50)
        ]
        
        for optimizer in optimizers:
            result = optimizer.optimize(
                objective_function=rastrigin_function,
                max_evaluations=2000
            )
            
            # Should find reasonable solution even in high dimensions
            assert result.best_score < 50.0
            
            # Check all parameters are within bounds
            for param, value in result.best_params.items():
                assert -5 <= value <= 5
    
    def test_multi_modal_performance(self, ackley_function, standard_search_space):
        """Test performance on multi-modal functions."""
        # Genetic algorithm should handle multi-modal well
        genetic = GeneticOptimizer(
            standard_search_space,
            population_size=100,
            mutation_rate=0.1,
            crossover_rate=0.9
        )
        
        # Run multiple times to test consistency
        scores = []
        for seed in range(5):
            genetic.set_seed(seed)
            result = genetic.optimize(
                objective_function=ackley_function,
                max_evaluations=1000
            )
            scores.append(result.best_score)
        
        # Should consistently find near-global optimum
        assert np.mean(scores) < 1.0
        assert np.std(scores) < 0.5


class TestOptimizationRobustness:
    """Test robustness of optimization algorithms."""
    
    def test_noisy_objective_handling(self):
        """Test optimization with noisy objective function."""
        search_space = {'x': {'type': 'float', 'min': -5, 'max': 5}}
        
        # Objective with significant noise
        def noisy_objective(params):
            true_value = (params['x'] - 2) ** 2
            noise = np.random.normal(0, 0.5)
            return true_value + noise
        
        # Bayesian optimizer should handle noise well
        optimizer = BayesianOptimizer(
            search_space,
            initial_samples=30,
            acquisition_function='ucb',
            noise_level=0.5
        )
        
        result = optimizer.optimize(
            objective_function=noisy_objective,
            max_evaluations=200
        )
        
        # Should find approximate optimum despite noise
        assert abs(result.best_params['x'] - 2.0) < 0.5
    
    def test_discrete_parameter_optimization(self):
        """Test optimization with discrete parameters."""
        search_space = {
            'batch_size': {'type': 'int', 'min': 16, 'max': 512},
            'learning_rate': {'type': 'float', 'min': 0.0001, 'max': 0.1},
            'optimizer': {'type': 'categorical', 'values': ['adam', 'sgd', 'rmsprop']},
            'layers': {'type': 'int', 'min': 1, 'max': 10}
        }
        
        def ml_objective(params):
            # Simulate ML model performance
            score = 0.8
            
            # Optimal around batch_size=128
            score -= 0.001 * abs(params['batch_size'] - 128)
            
            # Optimal around lr=0.001
            score -= 10 * abs(np.log10(params['learning_rate']) - (-3))
            
            # Adam is best
            if params['optimizer'] == 'adam':
                score += 0.1
            elif params['optimizer'] == 'rmsprop':
                score += 0.05
            
            # 4-5 layers optimal
            score -= 0.05 * abs(params['layers'] - 4.5)
            
            return -score  # Minimize negative score
        
        optimizer = SmartSearchOptimizer(search_space)
        result = optimizer.optimize(
            objective_function=ml_objective,
            max_evaluations=300
        )
        
        assert result.best_params['optimizer'] == 'adam'
        assert 64 <= result.best_params['batch_size'] <= 256
        assert 3 <= result.best_params['layers'] <= 6
        assert -result.best_score > 0.85
    
    def test_constraint_satisfaction(self):
        """Test optimization with complex constraints."""
        search_space = {
            'x': {'type': 'float', 'min': 0, 'max': 10},
            'y': {'type': 'float', 'min': 0, 'max': 10},
            'z': {'type': 'float', 'min': 0, 'max': 10}
        }
        
        def objective(params):
            return -(params['x'] * params['y'] * params['z'])  # Maximize volume
        
        constraints = [
            lambda p: p['x'] + p['y'] + p['z'] <= 10,  # Sum constraint
            lambda p: p['x'] * p['y'] >= 5,  # Product constraint
            lambda p: p['z'] >= 0.5 * p['x']  # Ratio constraint
        ]
        
        optimizer = GeneticOptimizer(
            search_space,
            population_size=100,
            constraints=constraints
        )
        
        result = optimizer.optimize(
            objective_function=objective,
            max_evaluations=1000
        )
        
        # Check all constraints are satisfied
        for constraint in constraints:
            assert constraint(result.best_params), "Constraint violated"
        
        # Should find good solution
        assert -result.best_score > 20.0
    
    def test_parallel_evaluation_consistency(self):
        """Test consistency with parallel objective evaluation."""
        search_space = {
            f'x{i}': {'type': 'float', 'min': -1, 'max': 1}
            for i in range(5)
        }
        
        def objective(params):
            # Simulate expensive computation
            time.sleep(0.01)
            return sum(v ** 2 for v in params.values())
        
        # Test with parallel evaluation
        optimizer = MLOptimizer(
            search_space,
            parallel_evaluations=4,
            model_type='gaussian_process'
        )
        
        result_parallel = optimizer.optimize(
            objective_function=objective,
            max_evaluations=100,
            batch_size=4
        )
        
        # Test sequential for comparison
        optimizer_seq = MLOptimizer(
            search_space,
            parallel_evaluations=1,
            model_type='gaussian_process'
        )
        
        result_sequential = optimizer_seq.optimize(
            objective_function=objective,
            max_evaluations=100,
            batch_size=1
        )
        
        # Both should find similar quality solutions
        assert abs(result_parallel.best_score - result_sequential.best_score) < 0.1


class TestTradingSpecificOptimization:
    """Test optimization for trading-specific problems."""
    
    def test_sharpe_ratio_optimization(self):
        """Test optimizing Sharpe ratio with realistic constraints."""
        search_space = {
            'lookback_period': {'type': 'int', 'min': 10, 'max': 200},
            'entry_threshold': {'type': 'float', 'min': 0.5, 'max': 3.0},
            'exit_threshold': {'type': 'float', 'min': 0.1, 'max': 1.0},
            'stop_loss': {'type': 'float', 'min': 0.005, 'max': 0.05},
            'position_size': {'type': 'float', 'min': 0.1, 'max': 1.0}
        }
        
        def sharpe_objective(params):
            # Simulate backtesting and Sharpe ratio calculation
            lookback = params['lookback_period']
            entry = params['entry_threshold']
            exit_t = params['exit_threshold']
            stop = params['stop_loss']
            size = params['position_size']
            
            # Simulate returns based on parameters
            base_sharpe = 1.5
            
            # Optimal lookback around 50
            base_sharpe -= 0.005 * abs(lookback - 50)
            
            # Entry/exit relationship
            if entry > 2 * exit_t:
                base_sharpe += 0.2
            
            # Risk-reward tradeoff
            base_sharpe += 10 * stop * (1 - size)
            
            # Add realistic noise
            noise = np.random.normal(0, 0.1)
            
            return -(base_sharpe + noise)  # Maximize Sharpe
        
        optimizer = BayesianOptimizer(
            search_space,
            initial_samples=50,
            acquisition_function='ei'  # Expected improvement
        )
        
        result = optimizer.optimize(
            objective_function=sharpe_objective,
            max_evaluations=300,
            target_score=-2.0  # Target Sharpe of 2.0
        )
        
        assert -result.best_score > 1.8  # Sharpe > 1.8
        assert result.best_params['entry_threshold'] > result.best_params['exit_threshold']
    
    def test_latency_constrained_optimization(self):
        """Test optimization with latency constraints."""
        search_space = {
            'buffer_size': {'type': 'int', 'min': 10, 'max': 1000},
            'update_frequency': {'type': 'float', 'min': 0.001, 'max': 1.0},
            'parallel_workers': {'type': 'int', 'min': 1, 'max': 16},
            'batch_processing': {'type': 'categorical', 'values': ['enabled', 'disabled']}
        }
        
        def latency_objective(params):
            # Simulate latency and throughput
            buffer = params['buffer_size']
            freq = params['update_frequency']
            workers = params['parallel_workers']
            batch = params['batch_processing'] == 'enabled'
            
            # Calculate latency (ms)
            base_latency = 10
            latency = base_latency + 0.1 * buffer / workers
            if batch:
                latency += 5  # Batch processing adds latency
            latency += 20 * freq  # Higher frequency = more latency
            
            # Calculate throughput (trades/sec)
            throughput = workers * 100
            if batch:
                throughput *= 2
            throughput *= (1 - freq)  # Lower frequency = higher throughput
            
            # Combine objectives with latency constraint
            if latency > 50:  # Hard constraint: latency < 50ms
                return 1000  # Penalty
            
            # Maximize throughput while minimizing latency
            return latency / 50 - throughput / 1000
        
        optimizer = SmartSearchOptimizer(
            search_space,
            exploitation_rate=0.8  # Focus on good regions
        )
        
        result = optimizer.optimize(
            objective_function=latency_objective,
            max_evaluations=200
        )
        
        # Calculate final metrics
        buffer = result.best_params['buffer_size']
        workers = result.best_params['parallel_workers']
        freq = result.best_params['update_frequency']
        batch = result.best_params['batch_processing'] == 'enabled'
        
        latency = 10 + 0.1 * buffer / workers + (5 if batch else 0) + 20 * freq
        throughput = workers * 100 * (2 if batch else 1) * (1 - freq)
        
        assert latency < 50  # Meets latency constraint
        assert throughput > 500  # Good throughput
    
    def test_multi_asset_portfolio_optimization(self):
        """Test optimizing multi-asset portfolio allocation."""
        # 10 assets to allocate
        assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 
                  'JPM', 'BAC', 'GS', 'V', 'MA']
        
        search_space = {
            f'weight_{asset}': {'type': 'float', 'min': 0.0, 'max': 0.5}
            for asset in assets
        }
        
        # Add risk parameters
        search_space.update({
            'max_volatility': {'type': 'float', 'min': 0.1, 'max': 0.3},
            'rebalance_frequency': {'type': 'int', 'min': 1, 'max': 30}
        })
        
        def portfolio_objective(params):
            # Extract weights
            weights = {
                asset: params[f'weight_{asset}']
                for asset in assets
            }
            
            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            if total_weight == 0:
                return 1000  # Invalid portfolio
            
            weights = {k: v/total_weight for k, v in weights.items()}
            
            # Simulate portfolio metrics
            expected_returns = {
                'AAPL': 0.15, 'GOOGL': 0.12, 'MSFT': 0.14, 'AMZN': 0.16, 'TSLA': 0.25,
                'JPM': 0.10, 'BAC': 0.09, 'GS': 0.11, 'V': 0.13, 'MA': 0.12
            }
            
            volatilities = {
                'AAPL': 0.20, 'GOOGL': 0.18, 'MSFT': 0.17, 'AMZN': 0.22, 'TSLA': 0.40,
                'JPM': 0.15, 'BAC': 0.16, 'GS': 0.18, 'V': 0.15, 'MA': 0.14
            }
            
            # Calculate portfolio return and volatility
            portfolio_return = sum(
                weights[asset] * expected_returns[asset]
                for asset in assets
            )
            
            # Simplified volatility (ignoring correlations)
            portfolio_vol = np.sqrt(sum(
                (weights[asset] * volatilities[asset]) ** 2
                for asset in assets
            ))
            
            # Check volatility constraint
            if portfolio_vol > params['max_volatility']:
                return 1000  # Constraint violation
            
            # Sharpe ratio (assuming risk-free rate of 0.02)
            sharpe = (portfolio_return - 0.02) / portfolio_vol
            
            # Penalize too frequent rebalancing
            rebalance_cost = 0.001 * (31 - params['rebalance_frequency'])
            
            return -(sharpe - rebalance_cost)  # Maximize Sharpe
        
        # Use genetic algorithm for portfolio optimization
        optimizer = GeneticOptimizer(
            search_space,
            population_size=200,
            elite_size=20,
            mutation_rate=0.1
        )
        
        result = optimizer.optimize(
            objective_function=portfolio_objective,
            max_evaluations=2000
        )
        
        # Extract and check results
        weights = {
            asset: result.best_params[f'weight_{asset}']
            for asset in assets
        }
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        # Should have diversified portfolio
        assert max(normalized_weights.values()) < 0.4  # No single asset > 40%
        assert len([w for w in normalized_weights.values() if w > 0.05]) >= 4  # At least 4 assets > 5%
        assert -result.best_score > 0.5  # Sharpe > 0.5