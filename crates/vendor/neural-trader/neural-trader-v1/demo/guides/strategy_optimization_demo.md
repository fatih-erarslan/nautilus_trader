# Strategy Optimization Demo

## Strategy Comparison and Optimization

### Step 1: List All Strategies
```
Use tool: mcp__ai-news-trader__list_strategies
```

### Step 2: Compare Top Strategies
```
Use tool: mcp__ai-news-trader__get_strategy_comparison
Parameters:
  strategies: ["momentum_trading_optimized", "swing_trading_optimized", "mean_reversion_optimized"]
  metrics: ["sharpe_ratio", "total_return", "max_drawdown", "win_rate"]
```

### Step 3: Run Backtest
```
Use tool: mcp__ai-news-trader__run_backtest
Parameters:
  strategy: "momentum_trading_optimized"
  symbol: "NVDA"
  start_date: "2024-05-28"
  end_date: "2025-06-28"
  benchmark: "sp500"
  include_costs: true
  use_gpu: true
```

### Step 4: Get Adaptive Recommendation
```
Use tool: mcp__ai-news-trader__adaptive_strategy_selection
Parameters:
  symbol: "AAPL"
  auto_switch: false
```

## Expected Results:
- List of 4+ strategies with performance metrics
- Side-by-side comparison with Sharpe ratios
- Detailed backtest with monthly returns
- AI recommendation based on current conditions
