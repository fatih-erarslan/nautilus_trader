"""Optimize command implementation."""

import click
import json
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import concurrent.futures


def optimize_command(ctx, strategy: str, metrics: List[str], iterations: int,
                    constraints: Dict[str, Any], input_file: Optional[str],
                    output_file: Optional[str]):
    """Execute optimize command."""
    
    strategy_text = "all strategies" if strategy == "all" else f"{strategy} strategy"
    click.echo(f"Optimizing {strategy_text}")
    click.echo(f"Optimization metric{'s' if len(metrics) > 1 else ''}: {', '.join(metrics)}")
    click.echo(f"{iterations} iterations")
    
    if constraints:
        click.echo("Constraints:")
        for key, value in constraints.items():
            click.echo(f"  {key}: {value}")
    
    # Load input data if provided
    input_data = None
    if input_file:
        with open(input_file, 'r') as f:
            input_data = json.load(f)
        click.echo(f"Loaded input data from {input_file}")
    
    # Determine strategies to optimize
    strategies = ['momentum', 'swing', 'mirror'] if strategy == 'all' else [strategy]
    
    results = {
        'optimization_results': {},
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'iterations': iterations,
            'metrics': metrics,
            'constraints': constraints
        }
    }
    
    # Optimize each strategy
    for strat in strategies:
        click.echo(f"\nOptimizing {strat}...")
        opt_result = optimize_strategy(strat, metrics, iterations, constraints, input_data)
        results['optimization_results'][strat] = opt_result
    
    # Save results if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        click.echo(f"\nOptimization results saved to {output_file}")
    
    # Display summary
    _display_optimization_summary(results)


def optimize_strategy(strategy: str, metrics: List[str], iterations: int,
                     constraints: Dict[str, Any], input_data: Optional[Dict]) -> Dict[str, Any]:
    """Optimize a single strategy."""
    
    # Parameter ranges for each strategy
    param_ranges = {
        'momentum': {
            'lookback_period': (10, 100),
            'entry_threshold': (0.5, 0.9),
            'exit_threshold': (0.1, 0.5),
            'position_size': (0.01, 0.1)
        },
        'swing': {
            'ma_short': (5, 20),
            'ma_long': (20, 100),
            'rsi_oversold': (20, 40),
            'rsi_overbought': (60, 80),
            'stop_loss': (0.01, 0.05)
        },
        'mirror': {
            'min_position_size': (1000000, 10000000),
            'follow_delay_hours': (1, 48),
            'position_scale': (0.001, 0.05)
        }
    }
    
    params = param_ranges.get(strategy, param_ranges['momentum'])
    
    # Run optimization
    best_params = None
    best_score = -float('inf')
    history = []
    
    with tqdm(total=iterations, desc=f"Optimizing {strategy}") as pbar:
        for i in range(iterations):
            # Generate random parameters
            trial_params = {}
            for param, (min_val, max_val) in params.items():
                if isinstance(min_val, int):
                    trial_params[param] = np.random.randint(min_val, max_val)
                else:
                    trial_params[param] = np.random.uniform(min_val, max_val)
            
            # Evaluate parameters
            score = evaluate_parameters(strategy, trial_params, metrics, constraints, input_data)
            
            # Track history
            history.append({
                'iteration': i,
                'params': trial_params.copy(),
                'score': score
            })
            
            # Update best if better
            if score > best_score:
                best_score = score
                best_params = trial_params.copy()
            
            pbar.update(1)
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'history': history,
        'convergence': _calculate_convergence(history)
    }


def evaluate_parameters(strategy: str, params: Dict[str, Any], metrics: List[str],
                       constraints: Dict[str, Any], input_data: Optional[Dict]) -> float:
    """Evaluate parameter set and return score."""
    
    # Simulate strategy performance with given parameters
    # This is a mock implementation
    np.random.seed(hash(str(params)) % 2**32)
    
    # Calculate metrics
    metric_values = {}
    for metric in metrics:
        if metric == 'sharpe':
            metric_values['sharpe'] = np.random.uniform(0, 3)
        elif metric == 'returns':
            metric_values['returns'] = np.random.uniform(-0.2, 0.5)
        elif metric == 'sortino':
            metric_values['sortino'] = np.random.uniform(0, 4)
        elif metric == 'calmar':
            metric_values['calmar'] = np.random.uniform(0, 2)
    
    # Check constraints
    if constraints:
        if 'max_drawdown' in constraints:
            drawdown = np.random.uniform(0, 0.5)
            if drawdown > constraints['max_drawdown']:
                return -float('inf')  # Constraint violated
        
        if 'min_trades' in constraints:
            trades = np.random.randint(10, 200)
            if trades < constraints['min_trades']:
                return -float('inf')  # Constraint violated
    
    # Combine metrics into single score
    if len(metrics) == 1:
        score = metric_values[metrics[0]]
    else:
        # Multi-objective: average of normalized metrics
        score = np.mean(list(metric_values.values()))
    
    return score


def _calculate_convergence(history: List[Dict]) -> Dict[str, Any]:
    """Calculate convergence statistics."""
    scores = [h['score'] for h in history]
    
    # Find iteration where best score was found
    best_score = max(scores)
    best_iteration = next(i for i, s in enumerate(scores) if s == best_score)
    
    # Calculate improvement over iterations
    improvements = []
    current_best = -float('inf')
    for score in scores:
        if score > current_best:
            current_best = score
            improvements.append(current_best)
        else:
            improvements.append(current_best)
    
    return {
        'best_iteration': best_iteration,
        'final_score': scores[-1],
        'best_score': best_score,
        'improvement_ratio': (best_score - scores[0]) / abs(scores[0]) if scores[0] != 0 else 0
    }


def _display_optimization_summary(results: Dict[str, Any]):
    """Display optimization summary."""
    click.echo("\nOptimization Summary")
    click.echo("=" * 50)
    
    for strategy, opt_result in results['optimization_results'].items():
        click.echo(f"\nStrategy: {strategy}")
        click.echo("-" * 30)
        click.echo(f"Best Score: {opt_result['best_score']:.4f}")
        click.echo(f"Best Parameters:")
        for param, value in opt_result['best_params'].items():
            if isinstance(value, float):
                click.echo(f"  {param}: {value:.4f}")
            else:
                click.echo(f"  {param}: {value}")
        
        convergence = opt_result['convergence']
        click.echo(f"Found at iteration: {convergence['best_iteration']}")
        click.echo(f"Improvement ratio: {convergence['improvement_ratio']:.2%}")