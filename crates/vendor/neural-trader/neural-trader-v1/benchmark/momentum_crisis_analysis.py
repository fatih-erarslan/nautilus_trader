#!/usr/bin/env python3
"""Emergency momentum strategy crisis analysis"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.benchmarks.strategy_benchmark import StrategyBenchmark, StrategyProfiler
from src.config import ConfigManager
import json
import numpy as np

def analyze_momentum_crisis():
    """Perform forensic analysis on momentum strategy failure"""
    
    # Initialize benchmark
    config_manager = ConfigManager()
    benchmark = StrategyBenchmark(config_manager)
    profiler = StrategyProfiler(config_manager)
    
    print("MOMENTUM STRATEGY CRISIS ANALYSIS")
    print("=" * 60)
    print("Analyzing catastrophic -91.9% annual return...")
    print()
    
    # Run momentum strategy benchmark
    print("1. Running basic momentum benchmark (30 days)...")
    result = benchmark.benchmark_strategy('momentum', duration_days=30)
    
    print(f"\nBasic 30-day Results:")
    print(f"  Total Return: {result.total_return:.2%}")
    print(f"  Annualized Return: {result.annualized_return:.2%}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {result.max_drawdown:.2%}")
    print(f"  Win Rate: {result.win_rate:.2%}")
    print(f"  Total Trades: {result.total_trades}")
    
    # Run across different market conditions
    print("\n2. Testing across market conditions...")
    market_results = profiler.profile_strategy_across_conditions('momentum')
    
    for condition, cond_result in market_results.items():
        print(f"\n{condition.upper()}:")
        print(f"  Annual Return: {cond_result.annualized_return:.2%}")
        print(f"  Sharpe Ratio: {cond_result.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {cond_result.max_drawdown:.2%}")
        print(f"  Win Rate: {cond_result.win_rate:.2%}")
    
    # Compare with other strategies
    print("\n3. Comparing with successful strategies...")
    strategies = ['momentum', 'mirror', 'mean_reversion']
    comparison_results = benchmark.compare_strategies(strategies, duration_days=30)
    
    print("\nStrategy Comparison:")
    for strategy, result in comparison_results.items():
        print(f"\n{strategy.upper()}:")
        print(f"  Annual Return: {result.annualized_return:.2%}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {result.max_drawdown:.2%}")
        
    # Generate crisis report
    crisis_report = {
        "timestamp": "2025-06-23T21:00:00Z",
        "crisis_metrics": {
            "annual_return": result.annualized_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate
        },
        "market_condition_analysis": {
            condition: {
                "annual_return": res.annualized_return,
                "sharpe_ratio": res.sharpe_ratio,
                "max_drawdown": res.max_drawdown
            }
            for condition, res in market_results.items()
        },
        "comparison_with_successful": {
            strategy: {
                "annual_return": res.annualized_return,
                "sharpe_ratio": res.sharpe_ratio
            }
            for strategy, res in comparison_results.items()
        }
    }
    
    # Save crisis report
    with open('momentum_crisis_report.json', 'w') as f:
        json.dump(crisis_report, f, indent=2)
    
    print("\n" + "=" * 60)
    print("CRISIS REPORT SAVED: momentum_crisis_report.json")
    
    return crisis_report

if __name__ == "__main__":
    analyze_momentum_crisis()