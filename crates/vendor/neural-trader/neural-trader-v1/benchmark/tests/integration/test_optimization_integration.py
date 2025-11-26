"""
Optimization integration test suite for AI News Trading benchmark system.

This module tests the integration between optimization algorithms and benchmark components:
- Strategy parameter optimization
- Multi-objective optimization
- Convergence validation
- Performance-aware optimization
- Real-time optimization feedback
"""

import asyncio
import time
import pytest
from unittest.mock import Mock, patch
import numpy as np
import json
from pathlib import Path

from benchmark.src.optimization.optimizer import StrategyOptimizer
from benchmark.src.optimization.algorithms import *
from benchmark.src.optimization.objectives import *
from benchmark.src.benchmarks.runner import BenchmarkRunner
from benchmark.src.simulation.simulator import MarketSimulator
from benchmark.src.config import Config


class TestOptimizationBenchmarkIntegration:
    """Test optimization integration with benchmark system."""
    
    @pytest.fixture
    async def optimization_config(self):
        """Create optimization test configuration."""
        return {
            'optimization': {
                'algorithms': {
                    'bayesian': {
                        'enabled': True,
                        'n_initial_points': 5,
                        'acquisition_function': 'expected_improvement',
                        'max_iterations': 50
                    },
                    'genetic': {
                        'enabled': True,
                        'population_size': 20,
                        'generations': 25,
                        'mutation_rate': 0.1,
                        'crossover_rate': 0.8
                    },
                    'grid_search': {
                        'enabled': True,
                        'parallel': True,
                        'max_combinations': 100
                    }
                },
                'objectives': {
                    'primary': 'sharpe_ratio',
                    'secondary': ['total_return', 'max_drawdown'],
                    'constraints': {
                        'max_drawdown': 0.2,
                        'min_trades': 10,
                        'max_latency_ms': 100
                    }
                },
                'parameters': {
                    'momentum': {
                        'lookback_period': {'type': 'int', 'range': [5, 50]},
                        'threshold': {'type': 'float', 'range': [0.01, 0.1]},
                        'position_size': {'type': 'float', 'range': [0.05, 0.3]}
                    },
                    'arbitrage': {
                        'min_spread': {'type': 'float', 'range': [0.0001, 0.01]},
                        'timeout': {'type': 'int', 'range': [1, 10]},
                        'max_position': {'type': 'float', 'range': [0.1, 0.5]}
                    },
                    'news_sentiment': {
                        'sentiment_threshold': {'type': 'float', 'range': [0.3, 0.9]},
                        'impact_decay': {'type': 'int', 'range': [300, 7200]},
                        'position_size': {'type': 'float', 'range': [0.05, 0.25]}
                    }
                }
            },
            'benchmark': {
                'optimization_suite': 'comprehensive',
                'validation_runs': 3,
                'performance_targets': {
                    'latency_p99': 100,
                    'throughput': 1000,
                    'memory_gb': 2
                }
            }
        }
    
    @pytest.fixture
    async def optimization_system(self, optimization_config):
        """Create integrated optimization system."""
        system = {
            'optimizer': StrategyOptimizer(optimization_config['optimization']),
            'benchmark_runner': BenchmarkRunner(optimization_config['benchmark']),
            'simulator': MarketSimulator()
        }
        
        # Initialize components
        await system['optimizer'].initialize()
        await system['benchmark_runner'].initialize()
        await system['simulator'].initialize()
        
        yield system
        
        # Cleanup
        await system['optimizer'].cleanup()
        await system['benchmark_runner'].cleanup()
        await system['simulator'].cleanup()
    
    @pytest.mark.asyncio
    async def test_bayesian_optimization_integration(self, optimization_system, optimization_config):
        """Test Bayesian optimization integrated with benchmark system."""
        optimizer = optimization_system['optimizer']
        benchmark_runner = optimization_system['benchmark_runner']
        
        # Configure Bayesian optimization
        bayesian_config = optimization_config['optimization']['algorithms']['bayesian']
        
        # Define objective function that uses benchmark results
        async def benchmark_objective(params):
            # Configure benchmark with optimized parameters
            await benchmark_runner.configure_strategy_parameters('momentum', params)
            
            # Run benchmark suite
            results = await benchmark_runner.run_suite('optimization')
            
            # Extract performance metrics
            performance = results['strategies']['momentum']
            
            # Multi-objective scoring
            sharpe = performance.get('sharpe_ratio', 0)
            returns = performance.get('total_return', 0)
            drawdown = performance.get('max_drawdown', 1)
            latency = performance.get('latency_p99', 1000)
            
            # Penalize constraint violations
            penalty = 0
            if drawdown > 0.2:
                penalty += (drawdown - 0.2) * 10
            if latency > 100:
                penalty += (latency - 100) * 0.01
            
            # Combined objective (maximize)
            objective_score = sharpe + returns * 0.5 - penalty
            
            return {
                'score': objective_score,
                'sharpe_ratio': sharpe,
                'total_return': returns,
                'max_drawdown': drawdown,
                'latency_p99': latency,
                'constraint_violations': penalty > 0
            }
        
        # Run Bayesian optimization
        optimization_results = await optimizer.optimize_bayesian(
            objective_function=benchmark_objective,
            parameter_space=optimization_config['optimization']['parameters']['momentum'],
            max_iterations=bayesian_config['max_iterations'],
            n_initial_points=bayesian_config['n_initial_points']
        )
        
        # Validate optimization results
        assert optimization_results['status'] == 'success'
        assert optimization_results['convergence_achieved']
        assert optimization_results['best_score'] > 0
        assert 'best_parameters' in optimization_results
        assert 'optimization_history' in optimization_results
        
        # Validate best parameters are within bounds
        best_params = optimization_results['best_parameters']
        param_config = optimization_config['optimization']['parameters']['momentum']
        
        for param_name, value in best_params.items():
            param_range = param_config[param_name]['range']
            assert param_range[0] <= value <= param_range[1], \
                f"Parameter {param_name}={value} outside range {param_range}"
        
        # Validate performance improvement
        initial_score = optimization_results['optimization_history'][0]['score']
        final_score = optimization_results['best_score']
        improvement = (final_score - initial_score) / abs(initial_score) if initial_score != 0 else 0
        
        assert improvement > 0.1, f"Optimization improvement {improvement:.1%} < 10%"
    
    @pytest.mark.asyncio
    async def test_genetic_algorithm_integration(self, optimization_system, optimization_config):
        """Test genetic algorithm optimization integrated with benchmarks."""
        optimizer = optimization_system['optimizer']
        benchmark_runner = optimization_system['benchmark_runner']
        
        # Configure genetic algorithm
        genetic_config = optimization_config['optimization']['algorithms']['genetic']
        
        # Define fitness function
        async def fitness_function(individual_params):
            # Test multiple strategies with these parameters
            strategies = ['momentum', 'arbitrage']
            total_fitness = 0
            
            for strategy in strategies:
                await benchmark_runner.configure_strategy_parameters(strategy, individual_params.get(strategy, {}))
                
                # Run benchmark for this strategy
                results = await benchmark_runner.run_strategy_benchmark(strategy)
                
                # Calculate fitness components
                sharpe = results.get('sharpe_ratio', 0)
                returns = results.get('total_return', 0)
                drawdown = abs(results.get('max_drawdown', 0))
                
                # Fitness score (higher is better)
                strategy_fitness = sharpe * 2 + returns - drawdown * 5
                total_fitness += strategy_fitness
            
            return total_fitness / len(strategies)
        
        # Run genetic optimization
        genetic_results = await optimizer.optimize_genetic(
            fitness_function=fitness_function,
            parameter_space=optimization_config['optimization']['parameters'],
            population_size=genetic_config['population_size'],
            generations=genetic_config['generations'],
            mutation_rate=genetic_config['mutation_rate'],
            crossover_rate=genetic_config['crossover_rate']
        )
        
        # Validate genetic algorithm results
        assert genetic_results['status'] == 'success'
        assert genetic_results['generations_completed'] > 0
        assert 'best_individual' in genetic_results
        assert 'population_history' in genetic_results
        assert 'fitness_evolution' in genetic_results
        
        # Check fitness improvement over generations
        fitness_history = genetic_results['fitness_evolution']
        initial_fitness = fitness_history[0]['max_fitness']
        final_fitness = fitness_history[-1]['max_fitness']
        
        assert final_fitness >= initial_fitness, "Genetic algorithm should not decrease fitness"
        
        # Validate convergence
        if len(fitness_history) > 10:
            recent_improvement = (fitness_history[-1]['max_fitness'] - fitness_history[-10]['max_fitness'])
            convergence_threshold = 0.01
            converged = recent_improvement < convergence_threshold
            
            if genetic_results['generations_completed'] >= genetic_config['generations']:
                assert converged or genetic_results['generations_completed'] == genetic_config['generations']
    
    @pytest.mark.asyncio
    async def test_multi_objective_optimization(self, optimization_system, optimization_config):
        """Test multi-objective optimization with Pareto front analysis."""
        optimizer = optimization_system['optimizer']
        benchmark_runner = optimization_system['benchmark_runner']
        
        # Define multiple objectives
        async def multi_objective_function(params):
            # Configure and run benchmark
            await benchmark_runner.configure_strategy_parameters('momentum', params)
            results = await benchmark_runner.run_suite('optimization')
            
            performance = results['strategies']['momentum']
            
            # Multiple objectives to optimize
            objectives = {
                'sharpe_ratio': performance.get('sharpe_ratio', 0),  # Maximize
                'total_return': performance.get('total_return', 0),  # Maximize
                'max_drawdown': -performance.get('max_drawdown', 0),  # Minimize (negate for maximization)
                'win_rate': performance.get('win_rate', 0),  # Maximize
                'profit_factor': performance.get('profit_factor', 1)  # Maximize
            }
            
            return objectives
        
        # Run multi-objective optimization
        pareto_results = await optimizer.optimize_multi_objective(
            objective_function=multi_objective_function,
            parameter_space=optimization_config['optimization']['parameters']['momentum'],
            population_size=30,
            generations=20
        )
        
        # Validate Pareto optimization results
        assert pareto_results['status'] == 'success'
        assert 'pareto_front' in pareto_results
        assert 'pareto_set' in pareto_results
        assert len(pareto_results['pareto_front']) > 0
        
        # Validate Pareto dominance
        pareto_front = pareto_results['pareto_front']
        
        for i, solution_a in enumerate(pareto_front):
            for j, solution_b in enumerate(pareto_front):
                if i != j:
                    # Check that no solution dominates another in the Pareto front
                    dominates = True
                    for objective in solution_a['objectives']:
                        if solution_a['objectives'][objective] <= solution_b['objectives'][objective]:
                            dominates = False
                            break
                    assert not dominates, "Pareto front contains dominated solutions"
        
        # Validate objective diversity
        objective_ranges = {}
        for objective in pareto_front[0]['objectives']:
            values = [sol['objectives'][objective] for sol in pareto_front]
            objective_ranges[objective] = max(values) - min(values)
        
        # Should have diversity in objectives
        for objective, range_val in objective_ranges.items():
            assert range_val > 0, f"No diversity in objective {objective}"
    
    @pytest.mark.asyncio
    async def test_performance_aware_optimization(self, optimization_system, optimization_config):
        """Test optimization that considers performance constraints."""
        optimizer = optimization_system['optimizer']
        benchmark_runner = optimization_system['benchmark_runner']
        
        # Define performance-aware objective
        async def performance_constrained_objective(params):
            # Configure strategy
            await benchmark_runner.configure_strategy_parameters('momentum', params)
            
            # Run performance benchmark
            perf_results = await benchmark_runner.run_performance_benchmark('momentum')
            
            # Extract performance metrics
            latency_p99 = perf_results.get('latency_p99', 1000)
            throughput = perf_results.get('throughput', 0)
            memory_usage = perf_results.get('memory_gb', 10)
            
            # Extract trading performance
            trading_results = await benchmark_runner.run_strategy_benchmark('momentum')
            sharpe = trading_results.get('sharpe_ratio', 0)
            returns = trading_results.get('total_return', 0)
            
            # Performance constraints
            performance_targets = optimization_config['benchmark']['performance_targets']
            
            # Penalty for constraint violations
            penalty = 0
            if latency_p99 > performance_targets['latency_p99']:
                penalty += (latency_p99 - performance_targets['latency_p99']) * 0.01
            
            if throughput < performance_targets['throughput']:
                penalty += (performance_targets['throughput'] - throughput) * 0.001
            
            if memory_usage > performance_targets['memory_gb']:
                penalty += (memory_usage - performance_targets['memory_gb']) * 2
            
            # Combined score
            performance_score = sharpe + returns - penalty
            
            return {
                'score': performance_score,
                'sharpe_ratio': sharpe,
                'total_return': returns,
                'latency_p99': latency_p99,
                'throughput': throughput,
                'memory_gb': memory_usage,
                'performance_penalty': penalty,
                'constraints_met': penalty == 0
            }
        
        # Run performance-aware optimization
        perf_results = await optimizer.optimize_with_constraints(
            objective_function=performance_constrained_objective,
            parameter_space=optimization_config['optimization']['parameters']['momentum'],
            max_iterations=30,
            constraint_tolerance=0.1
        )
        
        # Validate constraint satisfaction
        assert perf_results['status'] == 'success'
        best_result = perf_results['best_result']
        
        # Check performance constraints
        targets = optimization_config['benchmark']['performance_targets']
        assert best_result['latency_p99'] <= targets['latency_p99'] * 1.1, \
            f"Latency {best_result['latency_p99']}ms exceeds target {targets['latency_p99']}ms"
        assert best_result['throughput'] >= targets['throughput'] * 0.9, \
            f"Throughput {best_result['throughput']} below target {targets['throughput']}"
        assert best_result['memory_gb'] <= targets['memory_gb'] * 1.1, \
            f"Memory {best_result['memory_gb']}GB exceeds target {targets['memory_gb']}GB"
    
    @pytest.mark.asyncio
    async def test_optimization_convergence_analysis(self, optimization_system):
        """Test optimization convergence analysis and early stopping."""
        optimizer = optimization_system['optimizer']
        
        # Mock objective function with known optimum
        optimal_params = {'param_a': 0.5, 'param_b': 0.3, 'param_c': 0.8}
        
        async def mock_objective(params):
            # Quadratic function with known optimum
            score = 0
            for key, value in params.items():
                optimal_value = optimal_params[key]
                score -= (value - optimal_value) ** 2
            
            # Add some noise
            score += np.random.normal(0, 0.01)
            
            return {'score': score}
        
        # Parameter space
        param_space = {
            'param_a': {'type': 'float', 'range': [0, 1]},
            'param_b': {'type': 'float', 'range': [0, 1]},
            'param_c': {'type': 'float', 'range': [0, 1]}
        }
        
        # Run optimization with convergence analysis
        convergence_results = await optimizer.optimize_with_convergence_analysis(
            objective_function=mock_objective,
            parameter_space=param_space,
            max_iterations=100,
            convergence_threshold=0.001,
            patience=10
        )
        
        # Validate convergence
        assert convergence_results['status'] == 'success'
        assert 'convergence_analysis' in convergence_results
        
        convergence_info = convergence_results['convergence_analysis']
        assert 'converged' in convergence_info
        assert 'convergence_iteration' in convergence_info
        assert 'improvement_rate' in convergence_info
        
        # Check parameter accuracy
        best_params = convergence_results['best_parameters']
        for param_name, optimal_value in optimal_params.items():
            best_value = best_params[param_name]
            error = abs(best_value - optimal_value)
            assert error < 0.1, f"Parameter {param_name} error {error} > 0.1"
    
    @pytest.mark.asyncio
    async def test_real_time_optimization_feedback(self, optimization_system):
        """Test real-time optimization feedback and monitoring."""
        optimizer = optimization_system['optimizer']
        benchmark_runner = optimization_system['benchmark_runner']
        
        # Set up real-time monitoring
        optimization_progress = []
        
        async def progress_callback(iteration, best_score, current_params, elapsed_time):
            progress_info = {
                'iteration': iteration,
                'best_score': best_score,
                'parameters': current_params.copy(),
                'elapsed_time': elapsed_time,
                'timestamp': time.time()
            }
            optimization_progress.append(progress_info)
        
        # Define objective with realistic computation time
        async def realistic_objective(params):
            # Simulate benchmark computation time
            await asyncio.sleep(0.1)
            
            # Mock benchmark results
            base_score = np.random.uniform(0.5, 2.0)
            param_bonus = sum(params.values()) * 0.1
            
            return {'score': base_score + param_bonus}
        
        # Run optimization with real-time feedback
        param_space = {
            'param_1': {'type': 'float', 'range': [0, 1]},
            'param_2': {'type': 'float', 'range': [0, 1]}
        }
        
        realtime_results = await optimizer.optimize_with_feedback(
            objective_function=realistic_objective,
            parameter_space=param_space,
            max_iterations=20,
            progress_callback=progress_callback,
            feedback_interval=1
        )
        
        # Validate real-time feedback
        assert realtime_results['status'] == 'success'
        assert len(optimization_progress) > 0
        assert len(optimization_progress) <= 20
        
        # Check progress information
        for i, progress in enumerate(optimization_progress):
            assert progress['iteration'] == i + 1
            assert 'best_score' in progress
            assert 'parameters' in progress
            assert 'elapsed_time' in progress
            assert progress['elapsed_time'] > 0
        
        # Validate progress trend
        if len(optimization_progress) > 5:
            early_scores = [p['best_score'] for p in optimization_progress[:5]]
            late_scores = [p['best_score'] for p in optimization_progress[-5:]]
            
            avg_early = np.mean(early_scores)
            avg_late = np.mean(late_scores)
            
            # Should show improvement or stability
            assert avg_late >= avg_early * 0.95, "Optimization should not significantly degrade"
    
    @pytest.mark.asyncio
    async def test_parameter_sensitivity_analysis(self, optimization_system):
        """Test parameter sensitivity analysis during optimization."""
        optimizer = optimization_system['optimizer']
        
        # Define objective with known parameter sensitivities
        async def sensitivity_objective(params):
            # param_1 has high sensitivity (coefficient 5)
            # param_2 has medium sensitivity (coefficient 2)  
            # param_3 has low sensitivity (coefficient 0.5)
            score = (params['param_1'] * 5 + 
                    params['param_2'] * 2 + 
                    params['param_3'] * 0.5)
            
            return {'score': score}
        
        param_space = {
            'param_1': {'type': 'float', 'range': [0, 1]},
            'param_2': {'type': 'float', 'range': [0, 1]},
            'param_3': {'type': 'float', 'range': [0, 1]}
        }
        
        # Run optimization with sensitivity analysis
        sensitivity_results = await optimizer.optimize_with_sensitivity_analysis(
            objective_function=sensitivity_objective,
            parameter_space=param_space,
            max_iterations=50,
            sensitivity_samples=20
        )
        
        # Validate sensitivity analysis
        assert sensitivity_results['status'] == 'success'
        assert 'sensitivity_analysis' in sensitivity_results
        
        sensitivity_info = sensitivity_results['sensitivity_analysis']
        assert 'parameter_importance' in sensitivity_info
        assert 'sensitivity_scores' in sensitivity_info
        
        # Check parameter ranking
        importance = sensitivity_info['parameter_importance']
        
        # Should rank param_1 highest, param_3 lowest
        param_ranking = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        assert param_ranking[0][0] == 'param_1', "param_1 should have highest importance"
        assert param_ranking[-1][0] == 'param_3', "param_3 should have lowest importance"
        
        # Validate sensitivity ratios
        high_sensitivity = importance['param_1']
        low_sensitivity = importance['param_3']
        sensitivity_ratio = high_sensitivity / low_sensitivity if low_sensitivity > 0 else float('inf')
        
        assert sensitivity_ratio > 5, f"Sensitivity ratio {sensitivity_ratio} should reflect parameter differences"


if __name__ == '__main__':
    pytest.main([__file__])