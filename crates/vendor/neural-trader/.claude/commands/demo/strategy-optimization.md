# Claude Code Demo: Strategy Optimization

Learn how to compare, backtest, and optimize trading strategies using MCP tools.

## Strategy Discovery

### List Available Strategies
```
Show me all available trading strategies:
Use mcp__ai-news-trader__list_strategies

Explain each strategy's approach and show current performance metrics.
```

### Get Detailed Strategy Information
```
I want details about momentum trading:
Use mcp__ai-news-trader__get_strategy_info with:
- strategy: "momentum_trading_optimized"

Show parameters, optimization history, and recent performance.
```

## Strategy Comparison

### Basic Comparison
```
Compare the top trading strategies:
Use mcp__ai-news-trader__get_strategy_comparison with:
- strategies: ["momentum_trading_optimized", "swing_trading_optimized", "mean_reversion_optimized"]
- metrics: ["sharpe_ratio", "total_return", "max_drawdown", "win_rate"]

Create a comparison table and recommend the best overall strategy.
```

### Risk-Adjusted Comparison
```
Compare strategies focusing on risk metrics:
Use mcp__ai-news-trader__get_strategy_comparison with:
- strategies: ["momentum_trading_optimized", "swing_trading_optimized", "mean_reversion_optimized", "mirror_trading_optimized"]
- metrics: ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown", "var_95"]

Identify the best risk-adjusted strategy for conservative investors.
```

## Backtesting

### Single Strategy Backtest
```
Backtest momentum trading on NVIDIA:
Use mcp__ai-news-trader__run_backtest with:
- strategy: "momentum_trading_optimized"
- symbol: "NVDA"
- start_date: "2024-01-01"
- end_date: "2024-12-31"
- benchmark: "sp500"
- include_costs: true
- use_gpu: true

Show monthly returns, drawdown chart, and comparison to benchmark.
```

### Multi-Symbol Backtest
```
Test swing trading across different stocks:
For each symbol in [AAPL, MSFT, GOOGL, AMZN]:
Use mcp__ai-news-trader__run_backtest with:
- strategy: "swing_trading_optimized"
- start_date: "2024-06-01"
- end_date: "2024-12-31"
- include_costs: true
- use_gpu: true

Identify which symbols work best with swing trading.
```

### Strategy Stress Test
```
Stress test strategies across different market conditions:
Test each strategy during:
1. Bull market: 2023-01-01 to 2023-12-31
2. Volatile period: 2022-01-01 to 2022-12-31
3. Recent: 2024-01-01 to 2024-06-30

Compare how each strategy performs in different regimes.
```

## Parameter Optimization

### Basic Optimization
```
Optimize swing trading parameters:
Use mcp__ai-news-trader__optimize_strategy with:
- strategy: "swing_trading_optimized"
- symbol: "AAPL"
- parameter_ranges: {
    "rsi_period": [10, 20],
    "overbought": [65, 75],
    "oversold": [25, 35],
    "hold_days": [3, 10]
  }
- max_iterations: 500
- optimization_metric: "sharpe_ratio"
- use_gpu: true

Show the optimal parameters and expected improvement.
```

### Multi-Objective Optimization
```
Optimize for both returns and risk:
Use mcp__ai-news-trader__optimize_strategy with:
- strategy: "momentum_trading_optimized"
- symbol: "SPY"
- parameter_ranges: {
    "lookback_period": [10, 30],
    "momentum_threshold": [0.5, 2.0],
    "stop_loss": [0.02, 0.05],
    "take_profit": [0.05, 0.15]
  }
- optimization_metric: "sharpe_ratio"
- max_iterations: 1000
- use_gpu: true

Find parameters that maximize returns while minimizing drawdown.
```

## Adaptive Strategy Selection

### Market-Based Selection
```
Get strategy recommendation based on current market:
Use mcp__ai-news-trader__adaptive_strategy_selection with:
- symbol: "AAPL"
- auto_switch: false

Explain why the recommended strategy fits current conditions.
```

### Auto-Switch Testing
```
Test automatic strategy switching:
Use mcp__ai-news-trader__adaptive_strategy_selection with:
- symbol: "QQQ"
- auto_switch: true

Monitor which strategy is selected and track switching frequency.
```

## Advanced Workflows

### Portfolio Strategy Allocation
```
Optimize strategy mix for a portfolio:
1. Test each strategy on portfolio symbols
2. Find optimal allocation percentages
3. Consider correlation between strategies
4. Minimize overall portfolio risk

Recommend: X% momentum, Y% swing, Z% mean reversion
```

### Walk-Forward Analysis
```
Perform walk-forward optimization:
1. Optimize on 6 months of data
2. Test on next 2 months
3. Re-optimize and repeat
4. Track out-of-sample performance

This prevents overfitting and ensures robustness.
```

### Market Regime Detection
```
Develop regime-aware strategies:
1. Identify current market regime (trending/ranging/volatile)
2. Test which strategies work in each regime
3. Create switching rules based on regime
4. Backtest the meta-strategy

Show regime detection accuracy and switching performance.
```

## Performance Analysis

### Generate Performance Report
```
Create comprehensive performance report:
Use mcp__ai-news-trader__performance_report with:
- strategy: "momentum_trading_optimized"
- period_days: 90
- include_benchmark: true
- use_gpu: true

Include:
- Return statistics
- Risk metrics
- Trade analysis
- Monthly breakdown
- Benchmark comparison
```

### Strategy Health Check
```
Monitor strategy health:
Use mcp__ai-news-trader__monitor_strategy_health with:
- strategy: "swing_trading_optimized"

Identify any degradation in performance or parameter drift.
```

## Integration Examples

### News-Driven Strategy Selection
```
Select strategy based on news sentiment:
1. Check market sentiment with analyze_news
2. If sentiment > 0.5: Use momentum strategy
3. If sentiment < -0.5: Use mean reversion
4. If neutral: Use swing trading

Implement this logic and test effectiveness.
```

### Risk-Parity Strategy Mix
```
Create risk-balanced multi-strategy portfolio:
1. Calculate risk contribution of each strategy
2. Adjust allocations so each contributes equally to risk
3. Rebalance monthly based on performance
4. Track risk-adjusted returns

Show allocation percentages and expected Sharpe ratio.
```

## Output Templates

### Strategy Report Card
```
Strategy: Momentum Trading Optimized
Performance Period: 90 days
Total Return: +23.5%
Sharpe Ratio: 2.34
Max Drawdown: -8.7%
Win Rate: 58%
Profit Factor: 2.1
Best Month: +8.2%
Worst Month: -3.1%
Status: ✅ Healthy
```

### Optimization Results
```
Original Performance:
- Sharpe: 1.85
- Return: 18.5%
- Drawdown: -12.3%

Optimized Performance:
- Sharpe: 2.34 (+26%)
- Return: 23.5% (+27%)
- Drawdown: -8.7% (-29%)

Optimal Parameters:
- RSI Period: 14
- Overbought: 72
- Oversold: 28
- Hold Days: 5
```

## Best Practices

1. **Always include transaction costs** in backtests (0.1% default)
2. **Use at least 1 year of data** for reliable backtests
3. **Reserve 20% of data** for out-of-sample testing
4. **Optimize for Sharpe ratio**, not just returns
5. **Re-optimize monthly** as market conditions change
6. **Test multiple symbols** before deploying strategy
7. **Monitor live performance** vs backtest expectations

## Common Pitfalls

- **Overfitting**: Too many parameters or too little data
- **Survivorship bias**: Test on delisted symbols too
- **Look-ahead bias**: Ensure data availability in backtest
- **Transaction costs**: Small gains eaten by fees
- **Regime changes**: Strategy stops working in new market