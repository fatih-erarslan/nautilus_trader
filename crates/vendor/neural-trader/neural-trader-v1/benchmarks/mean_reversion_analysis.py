#!/usr/bin/env python3
"""
Focused analysis of mean reversion strategy performance.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add benchmark directory to path
sys.path.insert(0, str(Path(__file__).parent / 'benchmark/src'))

from benchmarks.strategy_benchmark import StrategyBenchmark, StrategyProfiler
from config import ConfigManager

def analyze_mean_reversion_performance():
    """Comprehensive analysis of mean reversion strategy performance."""
    
    print("=" * 80)
    print("MEAN REVERSION STRATEGY PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Initialize benchmark components
    config_manager = ConfigManager()
    benchmark = StrategyBenchmark(config_manager)
    profiler = StrategyProfiler(config_manager)
    
    # 1. Baseline Performance Analysis
    print("\n1. BASELINE PERFORMANCE ANALYSIS")
    print("-" * 40)
    
    baseline_result = benchmark.benchmark_strategy(
        strategy_name="mean_reversion",
        duration_days=365,
        initial_capital=100000.0
    )
    
    print(f"Strategy: Mean Reversion")
    print(f"Duration: 365 days")
    print(f"Initial Capital: $100,000")
    print(f"")
    print(f"PERFORMANCE METRICS:")
    print(f"  Total Return: {baseline_result.total_return:.3f} ({baseline_result.total_return*100:.1f}%)")
    print(f"  Annualized Return: {baseline_result.annualized_return:.3f} ({baseline_result.annualized_return*100:.1f}%)")
    print(f"  Sharpe Ratio: {baseline_result.sharpe_ratio:.3f}")
    print(f"  Sortino Ratio: {baseline_result.sortino_ratio:.3f}")
    print(f"  Max Drawdown: {baseline_result.max_drawdown:.3f} ({baseline_result.max_drawdown*100:.1f}%)")
    print(f"  Win Rate: {baseline_result.win_rate:.3f} ({baseline_result.win_rate*100:.1f}%)")
    print(f"  Profit Factor: {baseline_result.profit_factor:.3f}")
    print(f"  Total Trades: {baseline_result.total_trades}")
    print(f"  Avg Trade Duration: {baseline_result.avg_trade_duration:.1f} days")
    print(f"  Volatility: {baseline_result.volatility:.3f} ({baseline_result.volatility*100:.1f}%)")
    print(f"  Calmar Ratio: {baseline_result.calmar_ratio:.3f}")
    
    # 2. Market Condition Analysis
    print("\n2. MARKET CONDITION ANALYSIS")
    print("-" * 40)
    
    market_results = profiler.profile_strategy_across_conditions("mean_reversion")
    
    for condition, result in market_results.items():
        condition_name = condition.replace("mean_reversion_", "").upper()
        print(f"\n{condition_name} MARKET:")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.3f}")
        print(f"  Total Return: {result.total_return:.3f} ({result.total_return*100:.1f}%)")
        print(f"  Max Drawdown: {result.max_drawdown:.3f} ({result.max_drawdown*100:.1f}%)")
        print(f"  Win Rate: {result.win_rate:.3f} ({result.win_rate*100:.1f}%)")
        print(f"  Total Trades: {result.total_trades}")
    
    # 3. Comparative Analysis
    print("\n3. COMPARATIVE ANALYSIS")
    print("-" * 40)
    
    strategies = ['mean_reversion', 'momentum', 'mirror', 'swing']
    comparison_results = benchmark.compare_strategies(strategies, duration_days=365)
    
    print(f"{'Strategy':<15} {'Sharpe':<8} {'Return':<8} {'Drawdown':<10} {'Win Rate':<9} {'Trades':<7}")
    print("-" * 70)
    
    for strategy, result in comparison_results.items():
        print(f"{strategy:<15} {result.sharpe_ratio:<8.3f} {result.total_return*100:<8.1f}% "
              f"{result.max_drawdown*100:<10.1f}% {result.win_rate*100:<9.1f}% {result.total_trades:<7}")
    
    # 4. Parameter Analysis
    print("\n4. PARAMETER ANALYSIS")
    print("-" * 40)
    
    # Analyze different z-score thresholds
    print("\nZ-SCORE THRESHOLD ANALYSIS:")
    z_thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    for z_threshold in z_thresholds:
        # Temporarily modify the mean reversion implementation
        original_trades = benchmark._mean_reversion_strategy_trades
        
        def modified_mean_reversion_trades(price_data, initial_capital):
            """Modified mean reversion with custom z-threshold."""
            trades = []
            window = 50
            position = None
            position_size = 0.05
            
            for i in range(window, len(price_data) - 1):
                # Calculate moving average and z-score
                ma = price_data[i - window:i].mean()
                std = price_data[i - window:i].std()
                z_score = (price_data[i] - ma) / std if std > 0 else 0
                
                current_price = price_data[i]
                
                # Entry signals (extreme z-scores)
                if position is None:
                    if z_score > z_threshold:
                        # Price too high, short
                        quantity = (initial_capital * position_size) / current_price
                        position = {
                            'side': 'short',
                            'entry_time': i,
                            'entry_price': current_price,
                            'quantity': quantity,
                            'entry_z_score': z_score
                        }
                    elif z_score < -z_threshold:
                        # Price too low, long
                        quantity = (initial_capital * position_size) / current_price
                        position = {
                            'side': 'long',
                            'entry_time': i,
                            'entry_price': current_price,
                            'quantity': quantity,
                            'entry_z_score': z_score
                        }
                
                # Exit signals (return to mean)
                elif position is not None:
                    should_exit = False
                    
                    if position['side'] == 'long' and z_score > 0:
                        should_exit = True
                    elif position['side'] == 'short' and z_score < 0:
                        should_exit = True
                    
                    # Stop loss
                    if abs(z_score) > abs(position['entry_z_score']) * 1.5:
                        should_exit = True
                    
                    if should_exit:
                        # Calculate PnL
                        if position['side'] == 'long':
                            pnl = position['quantity'] * (current_price - position['entry_price'])
                            return_pct = (current_price - position['entry_price']) / position['entry_price']
                        else:
                            pnl = position['quantity'] * (position['entry_price'] - current_price)
                            return_pct = (position['entry_price'] - current_price) / position['entry_price']
                        
                        from benchmarks.strategy_benchmark import Trade
                        trade = Trade(
                            entry_time=position['entry_time'],
                            exit_time=i,
                            entry_price=position['entry_price'],
                            exit_price=current_price,
                            quantity=position['quantity'],
                            side=position['side'],
                            pnl=pnl,
                            return_pct=return_pct
                        )
                        trades.append(trade)
                        position = None
            
            return trades
        
        # Replace the method temporarily
        benchmark._mean_reversion_strategy_trades = modified_mean_reversion_trades
        
        # Run benchmark with this z-threshold
        result = benchmark.benchmark_strategy("mean_reversion", duration_days=365)
        
        print(f"  Z-Threshold {z_threshold:.1f}: Sharpe={result.sharpe_ratio:.3f}, Return={result.total_return*100:.1f}%, "
              f"Trades={result.total_trades}, Win Rate={result.win_rate*100:.1f}%")
        
        # Restore original method
        benchmark._mean_reversion_strategy_trades = original_trades
    
    # 5. Optimization Recommendations
    print("\n5. OPTIMIZATION RECOMMENDATIONS")
    print("-" * 40)
    
    # Calculate performance gaps
    best_strategy = max(comparison_results.items(), key=lambda x: x[1].sharpe_ratio)
    mean_rev_result = comparison_results['mean_reversion']
    
    sharpe_gap = best_strategy[1].sharpe_ratio - mean_rev_result.sharpe_ratio
    return_gap = best_strategy[1].total_return - mean_rev_result.total_return
    
    print(f"PERFORMANCE GAPS vs {best_strategy[0].upper()}:")
    print(f"  Sharpe Ratio Gap: {sharpe_gap:.3f} ({sharpe_gap/mean_rev_result.sharpe_ratio*100:.1f}% improvement needed)")
    print(f"  Return Gap: {return_gap:.3f} ({return_gap*100:.1f}% additional return needed)")
    
    print(f"\nCRITICAL OPTIMIZATION AREAS:")
    
    # Identify optimization priorities
    recommendations = []
    
    if mean_rev_result.win_rate < 0.6:
        recommendations.append(f"• LOW WIN RATE ({mean_rev_result.win_rate*100:.1f}%): Optimize entry signals for better trade quality")
    
    if mean_rev_result.profit_factor < 1.5:
        recommendations.append(f"• LOW PROFIT FACTOR ({mean_rev_result.profit_factor:.2f}): Improve exit timing and risk management")
    
    if mean_rev_result.max_drawdown > 0.15:
        recommendations.append(f"• HIGH DRAWDOWN ({mean_rev_result.max_drawdown*100:.1f}%): Add position sizing and risk controls")
    
    if mean_rev_result.total_trades < 50:
        recommendations.append(f"• LOW TRADE FREQUENCY ({mean_rev_result.total_trades}): Consider more sensitive entry criteria")
    
    if mean_rev_result.avg_trade_duration > 10:
        recommendations.append(f"• LONG HOLD TIMES ({mean_rev_result.avg_trade_duration:.1f} days): Optimize exit signals")
    
    # Z-score analysis recommendations
    print("\nPARAMETER OPTIMIZATION TARGETS:")
    print(f"• Z-Score Threshold: Current 2.0 may be too conservative")
    print(f"• Window Length: Current 50 periods may not adapt to market regimes")
    print(f"• Position Size: Current 5% is very conservative vs Mirror's 8%")
    print(f"• Exit Logic: Simple mean crossing vs sophisticated exit timing")
    
    for rec in recommendations:
        print(rec)
    
    # 6. Summary
    print("\n6. EXECUTIVE SUMMARY")
    print("-" * 40)
    
    print(f"CURRENT STATE:")
    print(f"  • Sharpe Ratio: {mean_rev_result.sharpe_ratio:.3f} (Target: 3.0+)")
    print(f"  • Annual Return: {mean_rev_result.annualized_return*100:.1f}% (Target: 60-80%)")
    print(f"  • Max Drawdown: {mean_rev_result.max_drawdown*100:.1f}% (Target: <12%)")
    
    improvement_potential = (best_strategy[1].sharpe_ratio / mean_rev_result.sharpe_ratio - 1) * 100
    print(f"\nIMPROVEMENT POTENTIAL: {improvement_potential:.0f}% Sharpe ratio improvement possible")
    
    # Save results to memory format
    analysis_data = {
        "timestamp": datetime.now().isoformat(),
        "baseline_metrics": baseline_result.to_dict(),
        "market_conditions": {k: v.to_dict() for k, v in market_results.items()},
        "strategy_comparison": {k: v.to_dict() for k, v in comparison_results.items()},
        "optimization_targets": {
            "z_threshold": {"current": 2.0, "optimal_range": "1.5-2.5"},
            "window_length": {"current": 50, "optimal_range": "20-80"},
            "position_size": {"current": 0.05, "optimal_range": "0.08-0.12"},
            "exit_logic": {"current": "simple_mean_crossing", "target": "sophisticated_exits"}
        },
        "performance_gaps": {
            "sharpe_gap": sharpe_gap,
            "return_gap": return_gap,
            "improvement_potential_pct": improvement_potential
        }
    }
    
    # Save analysis results
    with open('/workspaces/ai-news-trader/mean_reversion_analysis_results.json', 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    print(f"\nAnalysis results saved to: mean_reversion_analysis_results.json")
    
    return analysis_data

if __name__ == "__main__":
    analyze_mean_reversion_performance()