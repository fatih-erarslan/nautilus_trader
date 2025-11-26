"""Benchmark the current swing trading strategy."""

import sys
import os
import json
import numpy as np
from datetime import datetime

# Add the necessary paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'benchmark/src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from benchmarks.strategy_benchmark import StrategyBenchmark

# Configuration
config = {
    'benchmark': {
        'duration_days': 365,
        'initial_capital': 100000,
        'market_conditions': ['bull', 'bear', 'sideways', 'volatile']
    }
}

def run_swing_benchmark():
    """Run comprehensive swing trading benchmark."""
    print("=== Current Swing Trading Strategy Benchmark ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("-" * 50)
    
    benchmark = StrategyBenchmark(config)
    
    # Test in different market conditions
    all_results = {}
    
    for condition in config['benchmark']['market_conditions']:
        print(f"\nTesting in {condition.upper()} market conditions...")
        
        # Generate market data for condition
        np.random.seed(hash(f'swing_{condition}') % 2**32)
        
        if condition == 'bull':
            mu = 0.001
            sigma = 0.015
        elif condition == 'bear':
            mu = -0.0008
            sigma = 0.025
        elif condition == 'sideways':
            mu = 0.0001
            sigma = 0.01
        elif condition == 'volatile':
            mu = 0.0002
            sigma = 0.04
        
        # Generate price data
        steps = config['benchmark']['duration_days']
        initial_price = 100.0
        dW = np.random.normal(0, 1, steps)
        log_returns = (mu - 0.5 * sigma**2) + sigma * dW
        log_prices = np.cumsum(log_returns)
        prices = initial_price * np.exp(log_prices)
        
        # Run benchmark
        result = benchmark.benchmark_strategy(
            'swing',
            price_data=prices,
            duration_days=config['benchmark']['duration_days'],
            initial_capital=config['benchmark']['initial_capital']
        )
        
        all_results[condition] = result.to_dict()
        
        # Display results
        print(f"  Total Return: {result.total_return:.2%}")
        print(f"  Annual Return: {result.annualized_return:.2%}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {result.max_drawdown:.2%}")
        print(f"  Win Rate: {result.win_rate:.2%}")
        print(f"  Total Trades: {result.total_trades}")
    
    # Calculate overall metrics
    print("\n" + "=" * 50)
    print("OVERALL PERFORMANCE SUMMARY")
    print("=" * 50)
    
    avg_annual_return = np.mean([r['annualized_return'] for r in all_results.values()])
    avg_sharpe = np.mean([r['sharpe_ratio'] for r in all_results.values()])
    avg_drawdown = np.mean([r['max_drawdown'] for r in all_results.values()])
    avg_win_rate = np.mean([r['win_rate'] for r in all_results.values()])
    
    print(f"Average Annual Return: {avg_annual_return:.2%}")
    print(f"Average Sharpe Ratio: {avg_sharpe:.2f}")
    print(f"Average Max Drawdown: {avg_drawdown:.2%}")
    print(f"Average Win Rate: {avg_win_rate:.2%}")
    
    # Save results
    results_file = 'swing_benchmark_current_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'market_conditions': all_results,
            'summary': {
                'avg_annual_return': avg_annual_return,
                'avg_sharpe_ratio': avg_sharpe,
                'avg_max_drawdown': avg_drawdown,
                'avg_win_rate': avg_win_rate
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return all_results

if __name__ == '__main__':
    run_swing_benchmark()