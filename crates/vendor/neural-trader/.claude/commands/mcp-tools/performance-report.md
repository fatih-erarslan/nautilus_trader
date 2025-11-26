# Performance Report MCP Tool

## Overview
The `mcp__ai-news-trader__performance_report` tool generates comprehensive performance analytics reports for trading strategies. It provides detailed metrics, attribution analysis, benchmark comparisons, and visual insights to evaluate strategy effectiveness and identify areas for improvement.

## Tool Specifications

### Tool Name
`mcp__ai-news-trader__performance_report`

### Purpose
- Generate detailed performance analytics for trading strategies
- Compare performance against benchmarks and peer strategies
- Provide attribution analysis and risk-adjusted metrics
- Identify performance drivers and detractors
- Create actionable insights for strategy improvement
- Track performance evolution over time

## Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `strategy` | string | Trading strategy name to analyze (e.g., "momentum", "mean_reversion", "neural_enhanced") |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `period_days` | integer | 30 | Analysis period in days (30, 90, 180, 365) |
| `include_benchmark` | boolean | true | Include benchmark comparison in report |
| `use_gpu` | boolean | false | Enable GPU acceleration for complex calculations |

## Return Value Structure

```json
{
  "report_summary": {
    "strategy": "momentum",
    "period": {
      "start_date": "2024-11-01",
      "end_date": "2024-12-01",
      "trading_days": 21,
      "calendar_days": 30
    },
    "performance_rating": "EXCELLENT",
    "confidence_level": 0.87
  },
  "performance_metrics": {
    "returns": {
      "total_return": 0.0823,
      "annualized_return": 0.9876,
      "daily_average": 0.0039,
      "winning_days": 14,
      "losing_days": 7,
      "win_rate": 0.667
    },
    "risk_adjusted": {
      "sharpe_ratio": 2.34,
      "sortino_ratio": 3.12,
      "calmar_ratio": 2.78,
      "information_ratio": 1.56,
      "treynor_ratio": 0.089
    },
    "risk_metrics": {
      "volatility": 0.1523,
      "downside_deviation": 0.0987,
      "max_drawdown": -0.0354,
      "var_95": -0.0234,
      "beta": 0.87,
      "tracking_error": 0.0678
    }
  },
  "benchmark_comparison": {
    "benchmark": "S&P 500",
    "benchmark_return": 0.0234,
    "alpha": 0.0589,
    "beta": 0.87,
    "correlation": 0.82,
    "outperformance": 0.0589,
    "up_capture": 1.23,
    "down_capture": 0.76
  },
  "trade_analysis": {
    "total_trades": 42,
    "profitable_trades": 28,
    "average_trade": {
      "return": 0.00196,
      "duration_hours": 18.5,
      "profit_factor": 1.67
    },
    "best_trade": {
      "symbol": "AAPL",
      "return": 0.0423,
      "date": "2024-11-15"
    },
    "worst_trade": {
      "symbol": "TSLA",
      "return": -0.0287,
      "date": "2024-11-22"
    },
    "trade_distribution": {
      "by_hour": {...},
      "by_day_of_week": {...},
      "by_symbol": {...}
    }
  },
  "attribution_analysis": {
    "return_sources": {
      "market_timing": 0.0234,
      "stock_selection": 0.0456,
      "position_sizing": 0.0133,
      "total": 0.0823
    },
    "sector_contribution": {
      "technology": 0.0456,
      "healthcare": 0.0234,
      "financials": 0.0089,
      "others": 0.0044
    },
    "factor_exposure": {
      "momentum": 0.67,
      "value": -0.23,
      "quality": 0.45,
      "low_volatility": -0.12
    }
  },
  "portfolio_evolution": {
    "daily_values": [...],
    "cumulative_returns": [...],
    "rolling_sharpe": [...],
    "exposure_timeline": {...}
  },
  "insights": {
    "strengths": [
      "Strong momentum capture in trending markets",
      "Excellent risk-adjusted returns",
      "Low correlation with benchmark"
    ],
    "weaknesses": [
      "Underperformance during market reversals",
      "High concentration in tech sector",
      "Elevated trading costs"
    ],
    "recommendations": [
      "Implement sector rotation limits",
      "Add mean reversion overlay for choppy markets",
      "Optimize trade frequency to reduce costs"
    ]
  },
  "statistical_analysis": {
    "return_distribution": {
      "skewness": 0.34,
      "kurtosis": 3.21,
      "jarque_bera_p": 0.089
    },
    "autocorrelation": {
      "lag_1": 0.12,
      "lag_5": -0.05,
      "ljung_box_p": 0.234
    },
    "regime_analysis": {
      "bull_market_performance": 0.1234,
      "bear_market_performance": -0.0234,
      "sideways_market_performance": 0.0567
    }
  },
  "costs_and_execution": {
    "total_costs": {
      "commission": 823.45,
      "slippage": 456.78,
      "spread": 234.56,
      "total": 1514.79
    },
    "cost_impact": -0.0121,
    "execution_quality": {
      "fill_rate": 0.987,
      "average_slippage_bps": 2.3,
      "implementation_shortfall": 0.0034
    }
  },
  "report_metadata": {
    "generated_at": "2024-12-01T15:30:00Z",
    "data_quality_score": 0.98,
    "completeness": 1.0,
    "execution_time_ms": 234
  }
}
```

## Advanced Usage Examples

### Basic Performance Report
```python
# Generate simple 30-day performance report
report = await mcp.call_tool(
    "mcp__ai-news-trader__performance_report",
    {
        "strategy": "momentum"
    }
)

print(f"Strategy Performance: {report['performance_metrics']['returns']['total_return']:.2%}")
print(f"Sharpe Ratio: {report['performance_metrics']['risk_adjusted']['sharpe_ratio']:.2f}")
```

### Comprehensive Annual Report
```python
# Generate detailed annual performance report
annual_report = await mcp.call_tool(
    "mcp__ai-news-trader__performance_report",
    {
        "strategy": "neural_enhanced",
        "period_days": 365,
        "include_benchmark": true,
        "use_gpu": true
    }
)

# Extract key insights
print("\n=== Annual Performance Summary ===")
print(f"Total Return: {annual_report['performance_metrics']['returns']['total_return']:.2%}")
print(f"vs Benchmark: {annual_report['benchmark_comparison']['outperformance']:.2%}")
print(f"Max Drawdown: {annual_report['performance_metrics']['risk_metrics']['max_drawdown']:.2%}")
print(f"Win Rate: {annual_report['performance_metrics']['returns']['win_rate']:.1%}")

print("\n=== Key Insights ===")
for strength in annual_report['insights']['strengths']:
    print(f"✓ {strength}")
```

### Multi-Strategy Comparison Report
```python
# Compare multiple strategies
strategies = ["momentum", "mean_reversion", "neural_enhanced", "pairs_trading"]
comparison_results = {}

for strategy in strategies:
    report = await mcp.call_tool(
        "mcp__ai-news-trader__performance_report",
        {
            "strategy": strategy,
            "period_days": 90,
            "include_benchmark": true
        }
    )
    
    comparison_results[strategy] = {
        "return": report['performance_metrics']['returns']['total_return'],
        "sharpe": report['performance_metrics']['risk_adjusted']['sharpe_ratio'],
        "max_dd": report['performance_metrics']['risk_metrics']['max_drawdown'],
        "win_rate": report['performance_metrics']['returns']['win_rate']
    }

# Create comparison table
print("\nStrategy Comparison (90 days)")
print("-" * 60)
print(f"{'Strategy':<20} {'Return':<10} {'Sharpe':<10} {'Max DD':<10} {'Win Rate':<10}")
print("-" * 60)
for strategy, metrics in comparison_results.items():
    print(f"{strategy:<20} {metrics['return']:>8.2%} {metrics['sharpe']:>8.2f} {metrics['max_dd']:>8.2%} {metrics['win_rate']:>8.1%}")
```

### Rolling Performance Analysis
```python
# Analyze performance over rolling windows
rolling_windows = [30, 60, 90, 180, 365]
rolling_performance = {}

for window in rolling_windows:
    report = await mcp.call_tool(
        "mcp__ai-news-trader__performance_report",
        {
            "strategy": "swing_trading",
            "period_days": window,
            "include_benchmark": true
        }
    )
    
    rolling_performance[window] = {
        "return": report['performance_metrics']['returns']['annualized_return'],
        "sharpe": report['performance_metrics']['risk_adjusted']['sharpe_ratio'],
        "vs_benchmark": report['benchmark_comparison']['outperformance']
    }

# Plot performance decay
for window, perf in rolling_performance.items():
    print(f"{window:3d} days: Return={perf['return']:6.2%}, Sharpe={perf['sharpe']:5.2f}, Alpha={perf['vs_benchmark']:6.2%}")
```

## Integration with Other Tools

### 1. Performance → Optimization Pipeline
```python
# Analyze current performance
performance = await mcp.call_tool(
    "mcp__ai-news-trader__performance_report",
    {
        "strategy": "momentum",
        "period_days": 90
    }
)

# If underperforming, trigger optimization
if performance['performance_metrics']['risk_adjusted']['sharpe_ratio'] < 1.0:
    print("Strategy underperforming, initiating optimization...")
    
    optimization = await mcp.call_tool(
        "mcp__ai-news-trader__optimize_strategy",
        {
            "strategy": "momentum",
            "symbol": "SPY",
            "parameter_ranges": {
                "lookback": [10, 50],
                "threshold": [0.01, 0.05]
            },
            "optimization_metric": "sharpe_ratio"
        }
    )
    
    # Re-evaluate performance with new parameters
    new_performance = await mcp.call_tool(
        "mcp__ai-news-trader__performance_report",
        {
            "strategy": "momentum",
            "period_days": 30
        }
    )
```

### 2. Risk-Adjusted Performance Analysis
```python
# Get performance report
performance = await mcp.call_tool(
    "mcp__ai-news-trader__performance_report",
    {
        "strategy": "portfolio_strategy",
        "period_days": 180
    }
)

# Get detailed risk analysis
portfolio = [
    {"symbol": "SPY", "weight": 0.4},
    {"symbol": "QQQ", "weight": 0.3},
    {"symbol": "TLT", "weight": 0.2},
    {"symbol": "GLD", "weight": 0.1}
]

risk_analysis = await mcp.call_tool(
    "mcp__ai-news-trader__risk_analysis",
    {
        "portfolio": portfolio,
        "time_horizon": 0.5  # 6 months
    }
)

# Combine insights
risk_adjusted_report = {
    "raw_return": performance['performance_metrics']['returns']['total_return'],
    "risk_adjusted_return": performance['performance_metrics']['risk_adjusted']['sharpe_ratio'],
    "downside_risk": risk_analysis['risk_metrics']['downside_risk']['downside_deviation'],
    "tail_risk": risk_analysis['monte_carlo_results']['tail_risk_metrics'],
    "risk_contribution": risk_analysis['risk_contribution']
}
```

### 3. Correlation with Market Regimes
```python
# Analyze performance across different market conditions
# First, get overall performance
overall_performance = await mcp.call_tool(
    "mcp__ai-news-trader__performance_report",
    {
        "strategy": "adaptive_momentum",
        "period_days": 365
    }
)

# Then analyze correlations with various assets
correlation_assets = ["SPY", "TLT", "GLD", "VIX", "DXY"]
correlation_analysis = await mcp.call_tool(
    "mcp__ai-news-trader__correlation_analysis",
    {
        "symbols": ["STRATEGY_RETURNS"] + correlation_assets,
        "period_days": 365
    }
)

# Identify regime sensitivities
regime_performance = overall_performance['statistical_analysis']['regime_analysis']
print(f"Bull Market Performance: {regime_performance['bull_market_performance']:.2%}")
print(f"Bear Market Performance: {regime_performance['bear_market_performance']:.2%}")
print(f"Sideways Market Performance: {regime_performance['sideways_market_performance']:.2%}")
```

## Performance Optimization Tips

### 1. Efficient Report Generation
```python
# Cache commonly used metrics
cached_reports = {}

async def get_cached_report(strategy, period_days):
    cache_key = f"{strategy}_{period_days}"
    
    if cache_key not in cached_reports:
        cached_reports[cache_key] = await mcp.call_tool(
            "mcp__ai-news-trader__performance_report",
            {
                "strategy": strategy,
                "period_days": period_days,
                "use_gpu": true  # Speed up complex calculations
            }
        )
    
    return cached_reports[cache_key]

# Use cached reports for dashboard
strategies = ["momentum", "mean_reversion", "neural_enhanced"]
for strategy in strategies:
    report = await get_cached_report(strategy, 30)
    print(f"{strategy}: {report['performance_metrics']['returns']['total_return']:.2%}")
```

### 2. Incremental Performance Updates
```python
# Update performance metrics incrementally
async def update_daily_performance(strategy):
    # Get yesterday's report
    yesterday_report = await mcp.call_tool(
        "mcp__ai-news-trader__performance_report",
        {
            "strategy": strategy,
            "period_days": 30
        }
    )
    
    # Get today's trades
    todays_trades = await mcp.call_tool(
        "mcp__ai-news-trader__get_portfolio_status",
        {
            "include_analytics": true
        }
    )
    
    # Calculate incremental update
    updated_metrics = {
        "total_return": yesterday_report['performance_metrics']['returns']['total_return'] + todays_trades['daily_return'],
        "trade_count": yesterday_report['trade_analysis']['total_trades'] + todays_trades['trades_today']
    }
    
    return updated_metrics
```

### 3. Parallel Report Generation
```python
# Generate reports for multiple timeframes in parallel
import asyncio

async def generate_all_reports(strategy):
    timeframes = [7, 30, 90, 180, 365]
    
    tasks = []
    for days in timeframes:
        task = mcp.call_tool(
            "mcp__ai-news-trader__performance_report",
            {
                "strategy": strategy,
                "period_days": days,
                "use_gpu": true
            }
        )
        tasks.append(task)
    
    reports = await asyncio.gather(*tasks)
    
    return dict(zip(timeframes, reports))

# Generate comprehensive view
all_reports = await generate_all_reports("neural_enhanced")
for days, report in all_reports.items():
    print(f"{days:3d} days: Sharpe={report['performance_metrics']['risk_adjusted']['sharpe_ratio']:5.2f}")
```

## Risk Management Best Practices

### 1. Performance Attribution Drill-Down
```python
# Detailed attribution analysis
report = await mcp.call_tool(
    "mcp__ai-news-trader__performance_report",
    {
        "strategy": "factor_based",
        "period_days": 90,
        "use_gpu": true
    }
)

# Analyze return sources
attribution = report['attribution_analysis']['return_sources']
print("\nReturn Attribution:")
for source, contribution in attribution.items():
    if source != "total":
        percentage = (contribution / attribution['total']) * 100
        print(f"{source}: {contribution:.3f} ({percentage:.1f}% of total)")

# Factor exposure analysis
factors = report['attribution_analysis']['factor_exposure']
print("\nFactor Exposures:")
for factor, exposure in factors.items():
    risk_level = "HIGH" if abs(exposure) > 0.5 else "MODERATE" if abs(exposure) > 0.2 else "LOW"
    print(f"{factor}: {exposure:+.2f} ({risk_level} exposure)")
```

### 2. Drawdown Analysis
```python
# Deep dive into drawdown periods
async def analyze_drawdowns(strategy, period_days=365):
    report = await mcp.call_tool(
        "mcp__ai-news-trader__performance_report",
        {
            "strategy": strategy,
            "period_days": period_days
        }
    )
    
    max_dd = report['performance_metrics']['risk_metrics']['max_drawdown']
    
    # Get detailed drawdown information
    if abs(max_dd) > 0.10:  # Significant drawdown
        print(f"WARNING: Significant drawdown detected: {max_dd:.2%}")
        
        # Analyze trades during drawdown
        worst_trades = []
        trade_analysis = report['trade_analysis']
        
        # Recommendations for drawdown mitigation
        recommendations = [
            "Implement stop-loss at 5% per position",
            "Reduce position sizes during high volatility",
            "Add defensive assets to portfolio",
            "Use options for downside protection"
        ]
        
        return {
            "max_drawdown": max_dd,
            "recovery_time_estimate": abs(max_dd) * 90,  # Rough estimate in days
            "mitigation_strategies": recommendations
        }
```

### 3. Performance Consistency Metrics
```python
# Analyze performance consistency
async def consistency_analysis(strategy):
    # Get reports for different periods
    periods = [30, 60, 90, 180]
    consistency_metrics = {}
    
    for period in periods:
        report = await mcp.call_tool(
            "mcp__ai-news-trader__performance_report",
            {
                "strategy": strategy,
                "period_days": period
            }
        )
        
        consistency_metrics[period] = {
            "sharpe": report['performance_metrics']['risk_adjusted']['sharpe_ratio'],
            "win_rate": report['performance_metrics']['returns']['win_rate'],
            "volatility": report['performance_metrics']['risk_metrics']['volatility']
        }
    
    # Calculate consistency scores
    sharpe_consistency = 1 - np.std([m['sharpe'] for m in consistency_metrics.values()]) / np.mean([m['sharpe'] for m in consistency_metrics.values()])
    win_rate_stability = min([m['win_rate'] for m in consistency_metrics.values()]) / max([m['win_rate'] for m in consistency_metrics.values()])
    
    return {
        "sharpe_consistency": sharpe_consistency,
        "win_rate_stability": win_rate_stability,
        "overall_consistency": (sharpe_consistency + win_rate_stability) / 2
    }
```

## Common Issues and Solutions

### Issue: Incomplete Data Periods
**Solution**: Handle missing data gracefully
```python
try:
    report = await mcp.call_tool(
        "mcp__ai-news-trader__performance_report",
        {
            "strategy": "new_strategy",
            "period_days": 90
        }
    )
except Exception as e:
    if "insufficient data" in str(e).lower():
        # Try shorter period
        report = await mcp.call_tool(
            "mcp__ai-news-trader__performance_report",
            {
                "strategy": "new_strategy",
                "period_days": 30
            }
        )
        print("Note: Using 30-day period due to limited data")
```

### Issue: Performance Metrics Distorted by Outliers
**Solution**: Use robust statistics and filtering
```python
# Get report with outlier handling
report = await mcp.call_tool(
    "mcp__ai-news-trader__performance_report",
    {
        "strategy": "high_frequency",
        "period_days": 30,
        "use_gpu": true  # Enables advanced statistical processing
    }
)

# Check for outliers in returns
stats = report['statistical_analysis']['return_distribution']
if stats['kurtosis'] > 5:  # Heavy tails
    print("Warning: Return distribution has heavy tails")
    print(f"Consider using Sortino ratio ({report['performance_metrics']['risk_adjusted']['sortino_ratio']:.2f}) instead of Sharpe")
```

### Issue: Benchmark Comparison Not Meaningful
**Solution**: Use appropriate benchmarks or peer comparison
```python
# Custom benchmark selection based on strategy
strategy_benchmarks = {
    "tech_momentum": "QQQ",
    "value_investing": "IWD",
    "small_cap": "IWM",
    "global_macro": "ACWI",
    "crypto_trading": "BTC-USD"
}

strategy = "tech_momentum"
appropriate_benchmark = strategy_benchmarks.get(strategy, "SPY")

# Generate report with appropriate benchmark
report = await mcp.call_tool(
    "mcp__ai-news-trader__performance_report",
    {
        "strategy": strategy,
        "period_days": 180,
        "include_benchmark": true
        # Benchmark selection handled internally
    }
)
```

## See Also
- [Run Backtest Tool](run-backtest.md) - Historical performance validation
- [Optimize Strategy Tool](optimize-strategy.md) - Performance improvement
- [Risk Analysis Tool](risk-analysis.md) - Risk-adjusted performance metrics
- [Run Benchmark Tool](run-benchmark.md) - Performance benchmarking