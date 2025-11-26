#!/bin/bash
# Claude Flow Command: Demo - Strategy Optimization

echo "🎯 Claude Flow: Strategy Optimization Demo"
echo "========================================="
echo ""
echo "Compare, backtest, and optimize trading strategies with ML."
echo ""

cat << 'EOF'
### Available Strategies
1. **Momentum Trading**: Trend-following with neural enhancement
2. **Swing Trading**: Multi-day positions with sentiment integration
3. **Mean Reversion**: Statistical arbitrage with ML signals
4. **Mirror Trading**: Institutional strategy replication

### Demo Workflow

#### Step 1: List All Strategies
```
Use: mcp__ai-news-trader__list_strategies
```
Returns:
- Strategy names and descriptions
- Current performance metrics
- GPU optimization status
- Last update timestamps

#### Step 2: Compare Strategies
```
Use: mcp__ai-news-trader__get_strategy_comparison
Parameters:
  strategies: [
    "momentum_trading_optimized",
    "swing_trading_optimized",
    "mean_reversion_optimized"
  ]
  metrics: [
    "sharpe_ratio",      # Risk-adjusted returns
    "total_return",      # Absolute performance
    "max_drawdown",      # Maximum loss
    "win_rate",          # Trade success rate
    "profit_factor"      # Win/loss ratio
  ]
```

#### Step 3: Run Historical Backtest
```
Use: mcp__ai-news-trader__run_backtest
Parameters:
  strategy: "momentum_trading_optimized"
  symbol: "NVDA"
  start_date: "2024-01-01"
  end_date: "2024-12-31"
  benchmark: "sp500"
  include_costs: true    # Transaction costs
  use_gpu: true         # 100x faster
```

#### Step 4: Optimize Parameters
```
Use: mcp__ai-news-trader__optimize_strategy
Parameters:
  strategy: "swing_trading_optimized"
  symbol: "AAPL"
  parameter_ranges: {
    "rsi_period": [10, 20],
    "overbought": [65, 75],
    "oversold": [25, 35],
    "hold_days": [3, 10]
  }
  max_iterations: 1000
  optimization_metric: "sharpe_ratio"
  use_gpu: true
```

#### Step 5: Adaptive Strategy Selection
```
Use: mcp__ai-news-trader__adaptive_strategy_selection
Parameters:
  symbol: "AAPL"
  auto_switch: true    # Enable automatic switching
```

### Performance Metrics Explained

**Sharpe Ratio**: Risk-adjusted returns (higher is better)
- Excellent: > 2.0
- Good: 1.0 - 2.0
- Acceptable: 0.5 - 1.0
- Poor: < 0.5

**Maximum Drawdown**: Largest peak-to-trough decline
- Low Risk: < 10%
- Moderate: 10-20%
- High Risk: > 20%

**Win Rate**: Percentage of profitable trades
- Trend Following: 40-50% typical
- Mean Reversion: 60-70% typical
- High Frequency: 50-55% typical

### Optimization Techniques

1. **Grid Search**: Exhaustive parameter testing
2. **Bayesian Optimization**: Smart parameter exploration
3. **Genetic Algorithms**: Evolution-based optimization
4. **Walk-Forward Analysis**: Out-of-sample validation

### Example: Complete Optimization Flow
```python
# 1. Initial strategy assessment
strategies = list_strategies()
comparison = compare_strategies(strategies, metrics)

# 2. Select best baseline
best_strategy = comparison.best_by("sharpe_ratio")

# 3. Optimize parameters
optimized = optimize_strategy(
    best_strategy,
    symbol="SPY",
    parameter_ranges=default_ranges,
    use_gpu=True
)

# 4. Validate with backtest
results = run_backtest(
    optimized.strategy,
    start_date="2023-01-01",
    end_date="2024-12-31"
)

# 5. Deploy if profitable
if results.sharpe_ratio > 1.5:
    switch_active_strategy(to_strategy=optimized.strategy)
```

### Advanced Features

**Multi-Objective Optimization**:
- Maximize returns while minimizing drawdown
- Balance win rate with profit factor
- Optimize for different market regimes

**Regime Detection**:
- Bull market parameters
- Bear market parameters
- Sideways market parameters
- Automatic switching based on conditions

**Monte Carlo Validation**:
- 10,000 random market scenarios
- Confidence intervals for metrics
- Stress testing edge cases

EOF

echo ""
echo "📈 Typical Improvements from Optimization:"
echo "- Sharpe Ratio: +0.3 to +0.8"
echo "- Max Drawdown: -2% to -5%"
echo "- Win Rate: +5% to +15%"
echo "- Annual Return: +10% to +30%"
echo ""
echo "💡 Pro Tips:"
echo "- Always include transaction costs in backtests"
echo "- Use walk-forward analysis to prevent overfitting"
echo "- Optimize for Sharpe ratio, not just returns"
echo "- Test across multiple market cycles"
echo ""
echo "📚 Full guide: /workspaces/ai-news-trader/demo/guides/strategy_optimization_demo.md"