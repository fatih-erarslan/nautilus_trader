# Run Backtest MCP Tool

## Overview
The `mcp__ai-news-trader__run_backtest` tool performs comprehensive historical backtesting of trading strategies with GPU acceleration. It validates strategy performance against historical data and benchmarks, providing detailed metrics and risk-adjusted returns.

## Tool Specifications

### Tool Name
`mcp__ai-news-trader__run_backtest`

### Purpose
- Validate trading strategies against historical market data
- Compare performance against benchmarks (S&P 500, NASDAQ, etc.)
- Calculate risk-adjusted metrics (Sharpe, Sortino, Calmar ratios)
- Identify drawdowns and risk periods
- Optimize strategy parameters based on historical performance

## Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `strategy` | string | Trading strategy name (e.g., "momentum", "mean_reversion", "neural_enhanced") |
| `symbol` | string | Stock symbol or asset to backtest (e.g., "AAPL", "TSLA", "BTC-USD") |
| `start_date` | string | Backtest start date in YYYY-MM-DD format |
| `end_date` | string | Backtest end date in YYYY-MM-DD format |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `benchmark` | string | "sp500" | Benchmark for comparison ("sp500", "nasdaq", "dow", "custom") |
| `include_costs` | boolean | true | Include trading costs (spread, commission, slippage) |
| `use_gpu` | boolean | true | Enable GPU acceleration for faster processing |

## Return Value Structure

```json
{
  "strategy": "momentum",
  "symbol": "AAPL",
  "period": {
    "start": "2024-01-01",
    "end": "2024-12-31",
    "trading_days": 252
  },
  "performance": {
    "total_return": 0.2543,
    "annualized_return": 0.2543,
    "volatility": 0.1832,
    "sharpe_ratio": 1.39,
    "sortino_ratio": 1.87,
    "calmar_ratio": 2.15,
    "max_drawdown": -0.1184,
    "win_rate": 0.5873,
    "profit_factor": 1.45
  },
  "benchmark_comparison": {
    "benchmark_return": 0.1823,
    "alpha": 0.0720,
    "beta": 0.92,
    "correlation": 0.85,
    "information_ratio": 0.76
  },
  "trade_statistics": {
    "total_trades": 147,
    "winning_trades": 86,
    "losing_trades": 61,
    "avg_win": 0.0134,
    "avg_loss": -0.0092,
    "largest_win": 0.0423,
    "largest_loss": -0.0287,
    "avg_holding_period": "3.2 days"
  },
  "risk_metrics": {
    "value_at_risk_95": -0.0234,
    "conditional_var_95": -0.0312,
    "downside_deviation": 0.0978,
    "ulcer_index": 0.0567,
    "recovery_time": "12 days"
  },
  "costs": {
    "total_commission": 1470.00,
    "total_slippage": 892.34,
    "total_spread_cost": 423.12,
    "impact_on_return": -0.0142
  },
  "execution_time": {
    "total_ms": 234,
    "gpu_speedup": "6.2x"
  }
}
```

## Advanced Usage Examples

### Basic Backtest
```python
# Simple momentum strategy backtest
result = await mcp.call_tool(
    "mcp__ai-news-trader__run_backtest",
    {
        "strategy": "momentum",
        "symbol": "AAPL",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31"
    }
)
```

### Multi-Asset Backtest with Custom Benchmark
```python
# Backtest portfolio strategy against NASDAQ
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
results = []

for symbol in symbols:
    result = await mcp.call_tool(
        "mcp__ai-news-trader__run_backtest",
        {
            "strategy": "neural_enhanced",
            "symbol": symbol,
            "start_date": "2023-01-01",
            "end_date": "2024-12-31",
            "benchmark": "nasdaq",
            "use_gpu": true
        }
    )
    results.append(result)
```

### Walk-Forward Analysis
```python
# Rolling window backtests for robustness
window_size = 365  # days
step_size = 30     # days

for start in pd.date_range("2022-01-01", "2024-01-01", freq="30D"):
    end = start + pd.Timedelta(days=window_size)
    
    result = await mcp.call_tool(
        "mcp__ai-news-trader__run_backtest",
        {
            "strategy": "mean_reversion",
            "symbol": "SPY",
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "include_costs": true
        }
    )
```

### Parameter Sensitivity Analysis
```python
# Test strategy across different market conditions
market_periods = [
    {"start": "2020-03-01", "end": "2020-06-30", "label": "COVID_crash"},
    {"start": "2021-01-01", "end": "2021-12-31", "label": "bull_market"},
    {"start": "2022-01-01", "end": "2022-12-31", "label": "bear_market"},
    {"start": "2023-06-01", "end": "2024-01-01", "label": "recovery"}
]

for period in market_periods:
    result = await mcp.call_tool(
        "mcp__ai-news-trader__run_backtest",
        {
            "strategy": "swing_trading",
            "symbol": "QQQ",
            "start_date": period["start"],
            "end_date": period["end"],
            "use_gpu": true
        }
    )
    print(f"{period['label']}: Sharpe={result['performance']['sharpe_ratio']:.2f}")
```

## Integration with Other Tools

### 1. Optimize Then Backtest
```python
# First optimize strategy parameters
optimization = await mcp.call_tool(
    "mcp__ai-news-trader__optimize_strategy",
    {
        "strategy": "momentum",
        "symbol": "AAPL",
        "parameter_ranges": {
            "lookback_period": [10, 50],
            "threshold": [0.01, 0.05]
        }
    }
)

# Then backtest with optimized parameters
backtest = await mcp.call_tool(
    "mcp__ai-news-trader__run_backtest",
    {
        "strategy": "momentum",
        "symbol": "AAPL",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31"
    }
)
```

### 2. Neural Forecast Enhanced Backtest
```python
# Generate neural forecasts
forecast = await mcp.call_tool(
    "mcp__ai-news-trader__neural_forecast",
    {
        "symbol": "TSLA",
        "horizon": 24,
        "use_gpu": true
    }
)

# Backtest neural-enhanced strategy
backtest = await mcp.call_tool(
    "mcp__ai-news-trader__run_backtest",
    {
        "strategy": "neural_enhanced",
        "symbol": "TSLA",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "use_gpu": true
    }
)
```

### 3. Risk-Adjusted Portfolio Backtest
```python
# Run backtests for portfolio components
symbols = ["SPY", "TLT", "GLD", "QQQ"]
backtests = {}

for symbol in symbols:
    backtests[symbol] = await mcp.call_tool(
        "mcp__ai-news-trader__run_backtest",
        {
            "strategy": "buy_and_hold",
            "symbol": symbol,
            "start_date": "2023-01-01",
            "end_date": "2024-12-31"
        }
    )

# Analyze portfolio risk
risk_analysis = await mcp.call_tool(
    "mcp__ai-news-trader__risk_analysis",
    {
        "portfolio": [
            {"symbol": "SPY", "weight": 0.4},
            {"symbol": "TLT", "weight": 0.3},
            {"symbol": "GLD", "weight": 0.2},
            {"symbol": "QQQ", "weight": 0.1}
        ]
    }
)
```

## Performance Optimization Tips

### 1. GPU Acceleration
- Always use `use_gpu: true` for backtests longer than 1 year
- GPU provides 6-10x speedup for complex strategies
- Batch multiple backtests for GPU efficiency

### 2. Data Optimization
```python
# Preload data for multiple backtests
symbols = ["AAPL", "MSFT", "GOOGL"]
date_range = {"start": "2020-01-01", "end": "2024-12-31"}

# Cache data in memory
for symbol in symbols:
    # First run loads and caches data
    await mcp.call_tool(
        "mcp__ai-news-trader__run_backtest",
        {
            "strategy": "momentum",
            "symbol": symbol,
            **date_range,
            "use_gpu": true
        }
    )
```

### 3. Parallel Processing
```python
# Run multiple backtests in parallel
import asyncio

async def parallel_backtests(symbols, strategy, date_range):
    tasks = []
    for symbol in symbols:
        task = mcp.call_tool(
            "mcp__ai-news-trader__run_backtest",
            {
                "strategy": strategy,
                "symbol": symbol,
                **date_range,
                "use_gpu": true
            }
        )
        tasks.append(task)
    
    return await asyncio.gather(*tasks)
```

## Risk Management Best Practices

### 1. Out-of-Sample Testing
```python
# Split data for in-sample optimization and out-of-sample validation
in_sample = {
    "start_date": "2022-01-01",
    "end_date": "2023-06-30"
}

out_of_sample = {
    "start_date": "2023-07-01",
    "end_date": "2024-12-31"
}

# Optimize on in-sample
optimization = await mcp.call_tool(
    "mcp__ai-news-trader__optimize_strategy",
    {
        "strategy": "mean_reversion",
        "symbol": "SPY",
        **in_sample
    }
)

# Validate on out-of-sample
validation = await mcp.call_tool(
    "mcp__ai-news-trader__run_backtest",
    {
        "strategy": "mean_reversion",
        "symbol": "SPY",
        **out_of_sample
    }
)
```

### 2. Monte Carlo Stress Testing
```python
# Run backtests with different cost assumptions
cost_scenarios = [
    {"commission": 0.001, "slippage": 0.0005},
    {"commission": 0.002, "slippage": 0.001},
    {"commission": 0.005, "slippage": 0.002}
]

for scenario in cost_scenarios:
    result = await mcp.call_tool(
        "mcp__ai-news-trader__run_backtest",
        {
            "strategy": "high_frequency",
            "symbol": "SPY",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "include_costs": true
            # Cost parameters applied internally
        }
    )
```

### 3. Drawdown Analysis
```python
# Focus on risk metrics during backtesting
result = await mcp.call_tool(
    "mcp__ai-news-trader__run_backtest",
    {
        "strategy": "trend_following",
        "symbol": "QQQ",
        "start_date": "2020-01-01",
        "end_date": "2024-12-31"
    }
)

# Analyze drawdown periods
if result["performance"]["max_drawdown"] < -0.20:
    print("WARNING: Strategy experienced >20% drawdown")
    print(f"Recovery time: {result['risk_metrics']['recovery_time']}")
    print(f"Ulcer index: {result['risk_metrics']['ulcer_index']}")
```

## Common Issues and Solutions

### Issue: Slow Backtest Performance
**Solution**: Enable GPU acceleration and reduce data granularity
```python
result = await mcp.call_tool(
    "mcp__ai-news-trader__run_backtest",
    {
        "strategy": "complex_ml",
        "symbol": "AAPL",
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
        "use_gpu": true  # Essential for long backtests
    }
)
```

### Issue: Overfitting to Historical Data
**Solution**: Use walk-forward analysis and multiple validation periods
```python
# Implement walk-forward validation
validation_periods = generate_walk_forward_periods(
    start="2020-01-01",
    end="2024-12-31",
    train_months=12,
    test_months=3
)
```

### Issue: Unrealistic Cost Assumptions
**Solution**: Always include costs and use conservative estimates
```python
result = await mcp.call_tool(
    "mcp__ai-news-trader__run_backtest",
    {
        "strategy": "scalping",
        "symbol": "ES",  # E-mini S&P futures
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "include_costs": true  # Critical for high-frequency strategies
    }
)
```

## See Also
- [Optimize Strategy Tool](optimize-strategy.md) - Parameter optimization
- [Performance Report Tool](performance-report.md) - Detailed analytics
- [Risk Analysis Tool](risk-analysis.md) - Portfolio risk assessment
- [Neural Backtest Tool](../neural-trader/mcp-tools-reference.md#neural_backtest) - AI-enhanced backtesting