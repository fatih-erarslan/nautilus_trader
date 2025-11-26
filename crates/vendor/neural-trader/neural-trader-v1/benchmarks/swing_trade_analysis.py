#!/usr/bin/env python3
"""Detailed swing trade analysis to diagnose failure patterns."""

import sys
sys.path.insert(0, 'benchmark')

from src.benchmarks.strategy_benchmark import StrategyBenchmark, Trade
from src.config import ConfigManager
import numpy as np
import json
from datetime import datetime


class SwingTradeAnalyzer:
    """Analyzer for diagnosing swing trading failures."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.benchmark = StrategyBenchmark(self.config_manager)
        
    def analyze_swing_trades_in_detail(self):
        """Perform detailed analysis of swing trades."""
        # Generate price data
        np.random.seed(42)  # For reproducibility
        price_data = self.benchmark._generate_synthetic_price_data(252)
        
        # Get swing trades
        trades = self.benchmark._swing_strategy_trades(price_data, 100000)
        
        analysis = {
            "total_trades": len(trades),
            "profitable_trades": [],
            "losing_trades": [],
            "trade_details": [],
            "patterns": {},
            "timing_analysis": {},
            "risk_analysis": {}
        }
        
        # Analyze each trade
        for i, trade in enumerate(trades):
            trade_detail = {
                "trade_number": i + 1,
                "entry_day": int(trade.entry_time),
                "exit_day": int(trade.exit_time),
                "holding_days": trade.exit_time - trade.entry_time,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "side": trade.side,
                "return_pct": trade.return_pct * 100,
                "pnl": trade.pnl
            }
            
            # Calculate entry conditions
            entry_idx = int(trade.entry_time)
            if entry_idx >= 30:
                # Moving averages at entry
                short_ma = np.mean(price_data[entry_idx-10:entry_idx])
                long_ma = np.mean(price_data[entry_idx-30:entry_idx])
                trade_detail["ma_crossover"] = (short_ma > long_ma)
                trade_detail["price_vs_short_ma"] = (trade.entry_price - short_ma) / short_ma * 100
                
                # RSI approximation
                rsi_approx = self._calculate_rsi_approx(price_data[entry_idx-14:entry_idx+1])
                trade_detail["entry_rsi"] = rsi_approx
                
                # Momentum
                momentum = (price_data[entry_idx] - price_data[entry_idx-5]) / price_data[entry_idx-5]
                trade_detail["entry_momentum"] = momentum * 100
            
            analysis["trade_details"].append(trade_detail)
            
            if trade.pnl > 0:
                analysis["profitable_trades"].append(trade_detail)
            else:
                analysis["losing_trades"].append(trade_detail)
        
        # Pattern analysis
        analysis["patterns"]["avg_holding_days"] = np.mean([t["holding_days"] for t in analysis["trade_details"]])
        analysis["patterns"]["avg_winning_return"] = np.mean([t["return_pct"] for t in analysis["profitable_trades"]]) if analysis["profitable_trades"] else 0
        analysis["patterns"]["avg_losing_return"] = np.mean([t["return_pct"] for t in analysis["losing_trades"]]) if analysis["losing_trades"] else 0
        
        # Entry timing analysis
        long_trades = [t for t in analysis["trade_details"] if t.get("ma_crossover") == True]
        short_trades = [t for t in analysis["trade_details"] if t.get("ma_crossover") == False]
        
        analysis["timing_analysis"]["long_trades_count"] = len(long_trades)
        analysis["timing_analysis"]["short_trades_count"] = len(short_trades)
        analysis["timing_analysis"]["long_win_rate"] = sum(1 for t in long_trades if t["return_pct"] > 0) / len(long_trades) if long_trades else 0
        analysis["timing_analysis"]["short_win_rate"] = sum(1 for t in short_trades if t["return_pct"] > 0) / len(short_trades) if short_trades else 0
        
        # Risk analysis
        returns = [t["return_pct"] for t in analysis["trade_details"]]
        analysis["risk_analysis"]["max_consecutive_losses"] = self._max_consecutive_losses(trades)
        analysis["risk_analysis"]["largest_single_loss"] = min(returns) if returns else 0
        analysis["risk_analysis"]["largest_single_gain"] = max(returns) if returns else 0
        analysis["risk_analysis"]["risk_reward_ratio"] = abs(analysis["patterns"]["avg_winning_return"] / analysis["patterns"]["avg_losing_return"]) if analysis["patterns"]["avg_losing_return"] != 0 else 0
        
        return analysis
    
    def _calculate_rsi_approx(self, prices):
        """Approximate RSI calculation."""
        if len(prices) < 2:
            return 50
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses) if np.mean(losses) > 0 else 0.001
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _max_consecutive_losses(self, trades):
        """Calculate maximum consecutive losses."""
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if trade.pnl < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive
    
    def compare_with_successful_strategies(self):
        """Compare swing trading patterns with successful strategies."""
        # Get mirror strategy trades for comparison
        price_data = self.benchmark._generate_synthetic_price_data(252)
        
        swing_trades = self.benchmark._swing_strategy_trades(price_data, 100000)
        mirror_trades = self.benchmark._mirror_strategy_trades(price_data, 100000)
        mean_rev_trades = self.benchmark._mean_reversion_strategy_trades(price_data, 100000)
        
        comparison = {
            "swing": self._analyze_strategy_pattern(swing_trades, "swing"),
            "mirror": self._analyze_strategy_pattern(mirror_trades, "mirror"),
            "mean_reversion": self._analyze_strategy_pattern(mean_rev_trades, "mean_reversion")
        }
        
        return comparison
    
    def _analyze_strategy_pattern(self, trades, strategy_name):
        """Analyze trading patterns for a strategy."""
        if not trades:
            return {"error": "No trades generated"}
        
        returns = [t.return_pct for t in trades]
        holding_periods = [t.exit_time - t.entry_time for t in trades]
        
        return {
            "strategy": strategy_name,
            "trade_count": len(trades),
            "avg_return_per_trade": np.mean(returns) * 100,
            "avg_holding_period": np.mean(holding_periods),
            "win_rate": sum(1 for r in returns if r > 0) / len(returns) * 100,
            "profit_factor": self._calculate_profit_factor(trades),
            "max_position_size": 0.06 if strategy_name == "swing" else (0.08 if strategy_name == "mirror" else 0.05)
        }
    
    def _calculate_profit_factor(self, trades):
        """Calculate profit factor."""
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def generate_failure_diagnosis(self, analysis):
        """Generate diagnosis of swing trading failures."""
        diagnosis = {
            "critical_issues": [],
            "comparison_insights": [],
            "optimization_targets": []
        }
        
        # Critical issue identification
        if analysis["patterns"]["avg_losing_return"] < -3:
            diagnosis["critical_issues"].append({
                "issue": "Large average losses",
                "impact": f"Average losing trade: {analysis['patterns']['avg_losing_return']:.1f}%",
                "cause": "Poor stop loss placement or late exits"
            })
        
        if analysis["risk_analysis"]["risk_reward_ratio"] < 1.5:
            diagnosis["critical_issues"].append({
                "issue": "Poor risk/reward ratio",
                "impact": f"Current ratio: {analysis['risk_analysis']['risk_reward_ratio']:.2f}",
                "cause": "Taking profits too early or losses too late"
            })
        
        win_rate = len(analysis["profitable_trades"]) / len(analysis["trade_details"]) * 100
        if win_rate < 50:
            diagnosis["critical_issues"].append({
                "issue": "Low win rate",
                "impact": f"Only {win_rate:.1f}% of trades profitable",
                "cause": "Poor entry signal quality or market timing"
            })
        
        # Generate optimization targets
        diagnosis["optimization_targets"] = [
            {
                "parameter": "position_size",
                "current": "6% per trade",
                "recommended": "2-3% per trade",
                "reason": "Reduce risk per trade to improve overall performance"
            },
            {
                "parameter": "entry_signals",
                "current": "MA crossover + RSI",
                "recommended": "Add volume confirmation and support/resistance levels",
                "reason": "Improve entry timing and reduce false signals"
            },
            {
                "parameter": "stop_loss_method",
                "current": "Fixed percentage",
                "recommended": "ATR-based dynamic stops",
                "reason": "Adapt to market volatility"
            },
            {
                "parameter": "holding_period",
                "current": f"{analysis['patterns']['avg_holding_days']:.1f} days average",
                "recommended": "5-10 days with clear exit rules",
                "reason": "Optimize for swing trading timeframe"
            }
        ]
        
        return diagnosis


def main():
    """Run comprehensive swing trading analysis."""
    analyzer = SwingTradeAnalyzer()
    
    print("=" * 60)
    print("SWING TRADING FAILURE ANALYSIS")
    print("=" * 60)
    
    # Detailed trade analysis
    print("\n1. ANALYZING INDIVIDUAL TRADES...")
    trade_analysis = analyzer.analyze_swing_trades_in_detail()
    
    print(f"Total trades: {trade_analysis['total_trades']}")
    print(f"Profitable trades: {len(trade_analysis['profitable_trades'])}")
    print(f"Losing trades: {len(trade_analysis['losing_trades'])}")
    print(f"Win rate: {len(trade_analysis['profitable_trades']) / trade_analysis['total_trades'] * 100:.1f}%")
    
    print("\n2. PATTERN ANALYSIS:")
    print(f"Average holding period: {trade_analysis['patterns']['avg_holding_days']:.1f} days")
    print(f"Average winning return: {trade_analysis['patterns']['avg_winning_return']:.2f}%")
    print(f"Average losing return: {trade_analysis['patterns']['avg_losing_return']:.2f}%")
    print(f"Risk/Reward ratio: {trade_analysis['risk_analysis']['risk_reward_ratio']:.2f}")
    
    print("\n3. TIMING ANALYSIS:")
    print(f"Long trades: {trade_analysis['timing_analysis']['long_trades_count']} (Win rate: {trade_analysis['timing_analysis']['long_win_rate']*100:.1f}%)")
    print(f"Short trades: {trade_analysis['timing_analysis']['short_trades_count']} (Win rate: {trade_analysis['timing_analysis']['short_win_rate']*100:.1f}%)")
    
    print("\n4. RISK METRICS:")
    print(f"Max consecutive losses: {trade_analysis['risk_analysis']['max_consecutive_losses']}")
    print(f"Largest single loss: {trade_analysis['risk_analysis']['largest_single_loss']:.1f}%")
    print(f"Largest single gain: {trade_analysis['risk_analysis']['largest_single_gain']:.1f}%")
    
    # Strategy comparison
    print("\n5. COMPARING WITH SUCCESSFUL STRATEGIES...")
    comparison = analyzer.compare_with_successful_strategies()
    
    print("\nStrategy Comparison:")
    print("-" * 50)
    for strategy, metrics in comparison.items():
        print(f"\n{strategy.upper()}:")
        print(f"  Trades: {metrics.get('trade_count', 0)}")
        print(f"  Avg return/trade: {metrics.get('avg_return_per_trade', 0):.2f}%")
        print(f"  Win rate: {metrics.get('win_rate', 0):.1f}%")
        print(f"  Profit factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"  Position size: {metrics.get('max_position_size', 0)*100:.0f}%")
    
    # Generate diagnosis
    print("\n6. FAILURE DIAGNOSIS:")
    diagnosis = analyzer.generate_failure_diagnosis(trade_analysis)
    
    print("\nCRITICAL ISSUES:")
    for issue in diagnosis["critical_issues"]:
        print(f"- {issue['issue']}: {issue['impact']}")
        print(f"  Cause: {issue['cause']}")
    
    print("\nOPTIMIZATION TARGETS:")
    for target in diagnosis["optimization_targets"]:
        print(f"\n- {target['parameter'].upper()}:")
        print(f"  Current: {target['current']}")
        print(f"  Recommended: {target['recommended']}")
        print(f"  Reason: {target['reason']}")
    
    # Save detailed results
    results = {
        "trade_analysis": trade_analysis,
        "comparison": comparison,
        "diagnosis": diagnosis,
        "timestamp": datetime.now().isoformat()
    }
    
    with open("swing_failure_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Detailed results saved to swing_failure_analysis.json")
    
    return results


if __name__ == "__main__":
    main()