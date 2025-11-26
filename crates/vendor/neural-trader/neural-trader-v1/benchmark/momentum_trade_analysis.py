#!/usr/bin/env python3
"""Detailed trade-by-trade momentum failure analysis"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.benchmarks.strategy_benchmark import StrategyBenchmark
from src.config import ConfigManager
import numpy as np
import json
from datetime import datetime

def analyze_momentum_trades():
    """Perform detailed trade-by-trade analysis"""
    
    config_manager = ConfigManager()
    benchmark = StrategyBenchmark(config_manager)
    
    print("MOMENTUM TRADE-BY-TRADE FAILURE ANALYSIS")
    print("=" * 80)
    
    # Generate synthetic bull market data for analysis
    np.random.seed(42)
    days = 252  # 1 year
    
    # Bull market: positive drift with moderate volatility
    mu = 0.001  # 0.1% daily = ~25% annual
    sigma = 0.015  # 1.5% daily volatility
    initial_price = 100.0
    
    # Generate price path
    dW = np.random.normal(0, 1, days)
    log_returns = (mu - 0.5 * sigma**2) + sigma * dW
    log_prices = np.cumsum(log_returns)
    bull_prices = initial_price * np.exp(log_prices)
    
    print(f"Bull Market Simulation:")
    print(f"  Start Price: ${initial_price:.2f}")
    print(f"  End Price: ${bull_prices[-1]:.2f}")
    print(f"  Total Return: {(bull_prices[-1]/initial_price - 1)*100:.1f}%")
    print()
    
    # Run momentum strategy and get trades
    trades = benchmark._momentum_strategy_trades(bull_prices, 100000)
    
    print(f"Total Trades Generated: {len(trades)}")
    print()
    
    # Analyze each trade
    losing_trades = []
    winning_trades = []
    momentum_traps = []
    
    for i, trade in enumerate(trades):
        if trade.return_pct < 0:
            losing_trades.append(trade)
            
            # Check if this was a momentum trap (bought high, sold low)
            entry_idx = int(trade.entry_time)
            exit_idx = int(trade.exit_time)
            
            # Look at momentum before entry
            lookback = 20
            if entry_idx >= lookback:
                pre_entry_momentum = (bull_prices[entry_idx] - bull_prices[entry_idx-lookback]) / bull_prices[entry_idx-lookback]
                post_entry_return = (bull_prices[exit_idx] - bull_prices[entry_idx]) / bull_prices[entry_idx]
                
                if pre_entry_momentum > 0.02 and post_entry_return < -0.02:
                    momentum_traps.append({
                        'trade': trade,
                        'pre_entry_momentum': pre_entry_momentum,
                        'post_entry_return': post_entry_return,
                        'entry_price': bull_prices[entry_idx],
                        'exit_price': bull_prices[exit_idx]
                    })
        else:
            winning_trades.append(trade)
    
    # Calculate statistics
    total_return = sum(t.return_pct for t in trades)
    avg_win = np.mean([t.return_pct for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t.return_pct for t in losing_trades]) if losing_trades else 0
    win_rate = len(winning_trades) / len(trades) if trades else 0
    
    print("TRADE STATISTICS:")
    print(f"  Winning Trades: {len(winning_trades)} ({win_rate*100:.1f}%)")
    print(f"  Losing Trades: {len(losing_trades)} ({(1-win_rate)*100:.1f}%)")
    print(f"  Average Win: {avg_win*100:.2f}%")
    print(f"  Average Loss: {avg_loss*100:.2f}%")
    print(f"  Total Return: {total_return*100:.2f}%")
    print()
    
    print(f"MOMENTUM TRAPS IDENTIFIED: {len(momentum_traps)}")
    print("(Bought after strong momentum, suffered immediate reversal)")
    print()
    
    # Show first 5 momentum traps
    for i, trap in enumerate(momentum_traps[:5]):
        trade = trap['trade']
        print(f"Trap #{i+1}:")
        print(f"  Pre-entry momentum: {trap['pre_entry_momentum']*100:.1f}%")
        print(f"  Entry: ${trap['entry_price']:.2f} (Day {int(trade.entry_time)})")
        print(f"  Exit: ${trap['exit_price']:.2f} (Day {int(trade.exit_time)})")
        print(f"  Loss: {trade.return_pct*100:.2f}%")
        print()
    
    # Analyze entry/exit timing
    print("ENTRY/EXIT TIMING ANALYSIS:")
    
    # Check if strategy is consistently late to trends
    late_entries = 0
    early_exits = 0
    
    for trade in trades:
        entry_idx = int(trade.entry_time)
        exit_idx = int(trade.exit_time)
        
        # Check if bought near local high
        if entry_idx > 5 and entry_idx < len(bull_prices) - 5:
            local_high = max(bull_prices[entry_idx-5:entry_idx+5])
            if abs(bull_prices[entry_idx] - local_high) / local_high < 0.01:
                late_entries += 1
        
        # Check if sold near local low
        if exit_idx > 5 and exit_idx < len(bull_prices) - 5:
            local_low = min(bull_prices[exit_idx-5:exit_idx+5])
            if abs(bull_prices[exit_idx] - local_low) / local_low < 0.01:
                early_exits += 1
    
    print(f"  Late entries (bought near highs): {late_entries} ({late_entries/len(trades)*100:.1f}%)")
    print(f"  Early exits (sold near lows): {early_exits} ({early_exits/len(trades)*100:.1f}%)")
    print()
    
    # Generate failure diagnosis
    failure_diagnosis = {
        "timestamp": datetime.now().isoformat(),
        "bull_market_performance": {
            "market_return": (bull_prices[-1]/initial_price - 1),
            "strategy_return": total_return,
            "underperformance": (bull_prices[-1]/initial_price - 1) - total_return
        },
        "trade_statistics": {
            "total_trades": len(trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": abs(avg_win * len(winning_trades)) / abs(avg_loss * len(losing_trades)) if losing_trades else 0
        },
        "failure_modes": {
            "momentum_traps": len(momentum_traps),
            "momentum_trap_rate": len(momentum_traps) / len(trades) if trades else 0,
            "late_entry_rate": late_entries / len(trades) if trades else 0,
            "early_exit_rate": early_exits / len(trades) if trades else 0
        },
        "key_issues": [
            "Buying at momentum peaks before reversals",
            "Stop losses triggered by normal volatility",
            "Exit signals too sensitive to short-term reversals",
            "Momentum threshold (2%) may be too low",
            "No trend strength validation"
        ],
        "emergency_recommendations": [
            "Increase momentum threshold from 2% to 5%+",
            "Add trend quality filters (ADX, moving averages)",
            "Implement dynamic stop losses based on volatility",
            "Add volume confirmation requirements",
            "Use multiple timeframe momentum confirmation"
        ]
    }
    
    # Save detailed diagnosis
    with open('momentum_failure_diagnosis.json', 'w') as f:
        json.dump(failure_diagnosis, f, indent=2)
    
    print("DIAGNOSIS COMPLETE")
    print("Report saved to: momentum_failure_diagnosis.json")
    
    return failure_diagnosis

if __name__ == "__main__":
    analyze_momentum_trades()