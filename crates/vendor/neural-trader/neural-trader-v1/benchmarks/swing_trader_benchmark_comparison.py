"""Compare original and optimized swing trading strategies."""

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

def run_swing_comparison():
    """Run comprehensive comparison between original and optimized swing strategies."""
    print("=== Swing Trading Strategy Comparison ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("-" * 60)
    
    benchmark = StrategyBenchmark(config)
    
    # Store results for both strategies
    all_results = {
        'original': {},
        'optimized': {}
    }
    
    for condition in config['benchmark']['market_conditions']:
        print(f"\n{'='*60}")
        print(f"Testing in {condition.upper()} market conditions")
        print(f"{'='*60}")
        
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
        
        # Test original strategy
        print(f"\nOriginal Swing Strategy:")
        original_result = benchmark.benchmark_strategy(
            'swing',
            price_data=prices,
            duration_days=config['benchmark']['duration_days'],
            initial_capital=config['benchmark']['initial_capital']
        )
        all_results['original'][condition] = original_result.to_dict()
        
        print(f"  Total Return: {original_result.total_return:.2%}")
        print(f"  Annual Return: {original_result.annualized_return:.2%}")
        print(f"  Sharpe Ratio: {original_result.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {original_result.max_drawdown:.2%}")
        print(f"  Win Rate: {original_result.win_rate:.2%}")
        print(f"  Total Trades: {original_result.total_trades}")
        
        # Test optimized strategy
        print(f"\nOptimized Swing Strategy:")
        optimized_result = benchmark.benchmark_strategy(
            'swing_optimized',
            price_data=prices,
            duration_days=config['benchmark']['duration_days'],
            initial_capital=config['benchmark']['initial_capital']
        )
        all_results['optimized'][condition] = optimized_result.to_dict()
        
        print(f"  Total Return: {optimized_result.total_return:.2%}")
        print(f"  Annual Return: {optimized_result.annualized_return:.2%}")
        print(f"  Sharpe Ratio: {optimized_result.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {optimized_result.max_drawdown:.2%}")
        print(f"  Win Rate: {optimized_result.win_rate:.2%}")
        print(f"  Total Trades: {optimized_result.total_trades}")
        
        # Show improvement
        print(f"\nImprovement Metrics:")
        print(f"  Return Improvement: {(optimized_result.annualized_return - original_result.annualized_return):.2%}")
        print(f"  Sharpe Improvement: {(optimized_result.sharpe_ratio - original_result.sharpe_ratio):.2f}")
        print(f"  Drawdown Reduction: {(original_result.max_drawdown - optimized_result.max_drawdown):.2%}")
    
    # Calculate overall performance
    print("\n" + "=" * 60)
    print("OVERALL PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Original averages
    orig_avg_return = np.mean([r['annualized_return'] for r in all_results['original'].values()])
    orig_avg_sharpe = np.mean([r['sharpe_ratio'] for r in all_results['original'].values()])
    orig_avg_dd = np.mean([r['max_drawdown'] for r in all_results['original'].values()])
    orig_avg_wr = np.mean([r['win_rate'] for r in all_results['original'].values()])
    
    # Optimized averages
    opt_avg_return = np.mean([r['annualized_return'] for r in all_results['optimized'].values()])
    opt_avg_sharpe = np.mean([r['sharpe_ratio'] for r in all_results['optimized'].values()])
    opt_avg_dd = np.mean([r['max_drawdown'] for r in all_results['optimized'].values()])
    opt_avg_wr = np.mean([r['win_rate'] for r in all_results['optimized'].values()])
    
    print("\nOriginal Strategy Averages:")
    print(f"  Average Annual Return: {orig_avg_return:.2%}")
    print(f"  Average Sharpe Ratio: {orig_avg_sharpe:.2f}")
    print(f"  Average Max Drawdown: {orig_avg_dd:.2%}")
    print(f"  Average Win Rate: {orig_avg_wr:.2%}")
    
    print("\nOptimized Strategy Averages:")
    print(f"  Average Annual Return: {opt_avg_return:.2%}")
    print(f"  Average Sharpe Ratio: {opt_avg_sharpe:.2f}")
    print(f"  Average Max Drawdown: {opt_avg_dd:.2%}")
    print(f"  Average Win Rate: {opt_avg_wr:.2%}")
    
    print("\nTOTAL IMPROVEMENT:")
    print(f"  Return Boost: {((opt_avg_return / orig_avg_return - 1) * 100):.1f}% increase")
    print(f"  Sharpe Improvement: {opt_avg_sharpe - orig_avg_sharpe:.2f} points")
    print(f"  Risk Reduction: {((1 - opt_avg_dd / orig_avg_dd) * 100):.1f}% lower drawdown")
    print(f"  Win Rate Gain: {(opt_avg_wr - orig_avg_wr) * 100:.1f} percentage points")
    
    # Highlight volatile market improvement
    print("\nVOLATILE MARKET TRANSFORMATION:")
    print(f"  Original: {all_results['original']['volatile']['annualized_return']:.2%} return")
    print(f"  Optimized: {all_results['optimized']['volatile']['annualized_return']:.2%} return")
    print(f"  Improvement: {(all_results['optimized']['volatile']['annualized_return'] - all_results['original']['volatile']['annualized_return']):.2%}")
    
    # Save results
    results_file = 'swing_benchmark_comparison_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
            'summary': {
                'original': {
                    'avg_annual_return': orig_avg_return,
                    'avg_sharpe_ratio': orig_avg_sharpe,
                    'avg_max_drawdown': orig_avg_dd,
                    'avg_win_rate': orig_avg_wr
                },
                'optimized': {
                    'avg_annual_return': opt_avg_return,
                    'avg_sharpe_ratio': opt_avg_sharpe,
                    'avg_max_drawdown': opt_avg_dd,
                    'avg_win_rate': opt_avg_wr
                },
                'improvements': {
                    'return_boost_pct': (opt_avg_return / orig_avg_return - 1) * 100,
                    'sharpe_improvement': opt_avg_sharpe - orig_avg_sharpe,
                    'drawdown_reduction_pct': (1 - opt_avg_dd / orig_avg_dd) * 100,
                    'win_rate_gain_pct': (opt_avg_wr - orig_avg_wr) * 100
                }
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return all_results

if __name__ == '__main__':
    run_swing_comparison()