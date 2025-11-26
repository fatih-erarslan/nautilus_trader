# MCP Trading Tools Complete Reference (21 Tools)

## Overview
The AI News Trader platform provides 21 advanced trading tools through the Model Context Protocol (MCP) server. These tools enable neural forecasting, trading strategy execution, analytics, and real-time news sentiment analysis.

## Quick Start
```bash
# Start MCP server
python src/mcp/mcp_server_enhanced.py

# Tools are accessed through Claude integration
# Use the mcp__ai-news-trader__ prefix for all tools
```

---

## Neural Forecasting Tools (6 tools)

### 1. neural_forecast
**Generate AI-powered price predictions with confidence intervals**

**Parameters:**
- `symbol` (string, required): Stock ticker symbol
- `horizon` (integer, required): Forecast horizon in hours (1-720)
- `model_id` (string, optional): Specific model ID to use
- `confidence_level` (number, optional, default: 0.95): Confidence interval level
- `use_gpu` (boolean, optional, default: true): Enable GPU acceleration

**Example:**
```python
# Generate 24-hour price forecast for Apple
result = mcp__ai-news-trader__neural_forecast(
    symbol="AAPL",
    horizon=24,
    confidence_level=0.95,
    use_gpu=True
)
```

**Expected Output:**
```json
{
    "symbol": "AAPL",
    "current_price": 185.32,
    "forecast": {
        "point_forecast": [186.45, 187.12, 186.89, ...],
        "lower_bound": [185.23, 185.67, 185.45, ...],
        "upper_bound": [187.67, 188.57, 188.33, ...],
        "timestamps": ["2025-01-10T10:00:00", "2025-01-10T11:00:00", ...]
    },
    "model_info": {
        "model_type": "NHITS",
        "accuracy_metrics": {"mae": 0.82, "rmse": 1.15},
        "inference_time_ms": 8.3
    }
}
```

**Best Practices:**
- Use shorter horizons (1-24h) for day trading
- Longer horizons (168-720h) for position trading
- Always check confidence intervals for risk assessment

### 2. neural_train
**Train custom neural forecasting models on your data**

**Parameters:**
- `data_path` (string, required): Path to training data CSV
- `model_type` (string, required): Model type ("nhits", "nbeats", "transformer")
- `epochs` (integer, optional, default: 100): Training epochs
- `batch_size` (integer, optional, default: 32): Batch size
- `learning_rate` (number, optional, default: 0.001): Learning rate
- `validation_split` (number, optional, default: 0.2): Validation data split
- `use_gpu` (boolean, optional, default: true): Enable GPU acceleration

**Example:**
```python
# Train NHITS model on custom trading data
result = mcp__ai-news-trader__neural_train(
    data_path="/data/trading/spy_5min.csv",
    model_type="nhits",
    epochs=200,
    batch_size=64,
    learning_rate=0.0005,
    use_gpu=True
)
```

**Expected Output:**
```json
{
    "model_id": "nhits_spy_20250110_143022",
    "training_results": {
        "final_loss": 0.0234,
        "val_loss": 0.0289,
        "training_time_seconds": 342.5,
        "epochs_completed": 200
    },
    "model_path": "/models/nhits_spy_20250110_143022.pkl",
    "performance": {
        "train_mae": 0.67,
        "val_mae": 0.89,
        "gpu_speedup": "62.5x"
    }
}
```

**Best Practices:**
- Use at least 2 years of historical data
- Include volume and technical indicators
- Validate on out-of-sample data
- Save model checkpoints during training

### 3. neural_evaluate
**Evaluate model performance on test data**

**Parameters:**
- `model_id` (string, required): Model ID to evaluate
- `test_data` (string, required): Path to test data
- `metrics` (array, optional, default: ["mae", "rmse", "mape", "r2_score"]): Metrics to calculate
- `use_gpu` (boolean, optional, default: true): Enable GPU acceleration

**Example:**
```python
# Evaluate model on recent market data
result = mcp__ai-news-trader__neural_evaluate(
    model_id="nhits_spy_20250110_143022",
    test_data="/data/test/spy_jan2025.csv",
    metrics=["mae", "rmse", "mape", "directional_accuracy"],
    use_gpu=True
)
```

**Expected Output:**
```json
{
    "model_id": "nhits_spy_20250110_143022",
    "evaluation_metrics": {
        "mae": 0.92,
        "rmse": 1.34,
        "mape": 0.0045,
        "directional_accuracy": 0.687,
        "inference_time_ms": 7.2
    },
    "test_period": {
        "start": "2025-01-01",
        "end": "2025-01-10",
        "samples": 11520
    },
    "recommendations": "Model performs well for short-term predictions"
}
```

**Best Practices:**
- Use recent data for evaluation
- Compare multiple metrics
- Test during different market conditions
- Evaluate regularly for model drift

### 4. neural_backtest
**Run historical backtests with neural predictions**

**Parameters:**
- `model_id` (string, required): Neural model to use
- `start_date` (string, required): Backtest start date (YYYY-MM-DD)
- `end_date` (string, required): Backtest end date
- `benchmark` (string, optional, default: "sp500"): Benchmark to compare
- `rebalance_frequency` (string, optional, default: "daily"): Rebalancing frequency
- `use_gpu` (boolean, optional, default: true): Enable GPU acceleration

**Example:**
```python
# Backtest neural trading strategy
result = mcp__ai-news-trader__neural_backtest(
    model_id="nhits_spy_20250110_143022",
    start_date="2024-01-01",
    end_date="2024-12-31",
    benchmark="sp500",
    rebalance_frequency="hourly",
    use_gpu=True
)
```

**Expected Output:**
```json
{
    "backtest_results": {
        "total_return": 0.287,
        "annualized_return": 0.312,
        "sharpe_ratio": 2.15,
        "max_drawdown": -0.078,
        "win_rate": 0.612
    },
    "benchmark_comparison": {
        "strategy_return": 0.287,
        "benchmark_return": 0.195,
        "excess_return": 0.092,
        "information_ratio": 1.85
    },
    "trade_statistics": {
        "total_trades": 1847,
        "winning_trades": 1130,
        "average_win": 0.0034,
        "average_loss": -0.0021
    }
}
```

**Best Practices:**
- Include transaction costs
- Test multiple market cycles
- Compare against buy-and-hold
- Validate with walk-forward analysis

### 5. neural_model_status
**Monitor model health and performance**

**Parameters:**
- `model_id` (string, optional): Specific model ID or null for all models

**Example:**
```python
# Check all neural models
result = mcp__ai-news-trader__neural_model_status()

# Check specific model
result = mcp__ai-news-trader__neural_model_status(
    model_id="nhits_spy_20250110_143022"
)
```

**Expected Output:**
```json
{
    "models": [
        {
            "model_id": "nhits_spy_20250110_143022",
            "status": "active",
            "last_prediction": "2025-01-10T15:32:10Z",
            "performance_24h": {
                "predictions": 1234,
                "avg_accuracy": 0.923,
                "avg_latency_ms": 8.7
            },
            "health_check": "healthy",
            "gpu_utilization": 0.45
        }
    ],
    "system_status": {
        "gpu_available": true,
        "total_models": 3,
        "active_models": 2
    }
}
```

**Best Practices:**
- Monitor model drift regularly
- Set accuracy thresholds
- Track inference latency
- Rotate models based on performance

### 6. neural_optimize
**Optimize model hyperparameters**

**Parameters:**
- `model_id` (string, required): Model to optimize
- `parameter_ranges` (object, required): Parameter search ranges
- `optimization_metric` (string, optional, default: "mae"): Metric to optimize
- `trials` (integer, optional, default: 100): Number of optimization trials
- `use_gpu` (boolean, optional, default: true): Enable GPU acceleration

**Example:**
```python
# Optimize NHITS hyperparameters
result = mcp__ai-news-trader__neural_optimize(
    model_id="nhits_base",
    parameter_ranges={
        "learning_rate": [0.0001, 0.01],
        "hidden_units": [64, 512],
        "dropout_rate": [0.1, 0.5],
        "batch_size": [16, 128]
    },
    optimization_metric="sharpe_ratio",
    trials=200,
    use_gpu=True
)
```

**Expected Output:**
```json
{
    "optimization_results": {
        "best_parameters": {
            "learning_rate": 0.0003,
            "hidden_units": 256,
            "dropout_rate": 0.2,
            "batch_size": 64
        },
        "best_score": 2.43,
        "improvement": 0.18,
        "trials_completed": 200
    },
    "performance_comparison": {
        "original_score": 2.05,
        "optimized_score": 2.43,
        "percentage_improvement": 18.5
    }
}
```

**Best Practices:**
- Use Bayesian optimization
- Cross-validate results
- Test on holdout data
- Document parameter choices

---

## Trading Strategy Tools (4 tools)

### 7. quick_analysis
**Get instant market analysis for any symbol**

**Parameters:**
- `symbol` (string, required): Stock ticker symbol
- `use_gpu` (boolean, optional, default: false): Enable GPU acceleration

**Example:**
```python
# Quick analysis of Tesla
result = mcp__ai-news-trader__quick_analysis(
    symbol="TSLA",
    use_gpu=True
)
```

**Expected Output:**
```json
{
    "symbol": "TSLA",
    "current_price": 234.56,
    "analysis": {
        "trend": "bullish",
        "strength": 0.72,
        "support": 225.00,
        "resistance": 245.00,
        "volume_trend": "increasing"
    },
    "indicators": {
        "rsi": 67.3,
        "macd": "bullish_crossover",
        "moving_averages": {
            "ma20": 228.45,
            "ma50": 215.23,
            "ma200": 198.67
        }
    },
    "recommendation": "buy",
    "confidence": 0.78
}
```

**Best Practices:**
- Combine with neural forecasts
- Check multiple timeframes
- Verify with volume analysis
- Use for entry/exit timing

### 8. simulate_trade
**Simulate trades with real-time performance tracking**

**Parameters:**
- `strategy` (string, required): Trading strategy name
- `symbol` (string, required): Stock ticker symbol
- `action` (string, required): Trade action ("buy", "sell", "short", "cover")
- `use_gpu` (boolean, optional, default: false): Enable GPU acceleration

**Example:**
```python
# Simulate momentum trade
result = mcp__ai-news-trader__simulate_trade(
    strategy="neural_momentum",
    symbol="SPY",
    action="buy",
    use_gpu=True
)
```

**Expected Output:**
```json
{
    "simulation_results": {
        "trade_id": "sim_20250110_154233",
        "symbol": "SPY",
        "action": "buy",
        "entry_price": 456.78,
        "position_size": 100,
        "strategy": "neural_momentum",
        "expected_outcomes": {
            "1h": {"price": 457.34, "pnl": 56.00},
            "4h": {"price": 458.12, "pnl": 134.00},
            "1d": {"price": 459.45, "pnl": 267.00}
        },
        "risk_metrics": {
            "var_95": -234.00,
            "stop_loss": 454.23
        }
    }
}
```

**Best Practices:**
- Test strategies before live trading
- Simulate various market conditions
- Track slippage and costs
- Validate against historical data

### 9. execute_trade
**Execute live trades with advanced order management**

**Parameters:**
- `strategy` (string, required): Trading strategy name
- `symbol` (string, required): Stock ticker symbol
- `action` (string, required): Trade action
- `quantity` (integer, required): Number of shares
- `order_type` (string, optional, default: "market"): Order type
- `limit_price` (number, optional): Limit price for limit orders

**Example:**
```python
# Execute limit order
result = mcp__ai-news-trader__execute_trade(
    strategy="mean_reversion",
    symbol="AAPL",
    action="buy",
    quantity=50,
    order_type="limit",
    limit_price=185.25
)
```

**Expected Output:**
```json
{
    "trade_execution": {
        "order_id": "ORD_20250110_154512",
        "status": "filled",
        "symbol": "AAPL",
        "action": "buy",
        "quantity": 50,
        "fill_price": 185.23,
        "commission": 1.00,
        "timestamp": "2025-01-10T15:45:12.345Z"
    },
    "position_update": {
        "symbol": "AAPL",
        "quantity": 150,
        "avg_price": 183.45,
        "unrealized_pnl": 267.00
    }
}
```

**Best Practices:**
- Use limit orders for better fills
- Implement stop-loss orders
- Monitor position sizes
- Track execution quality

### 10. get_portfolio_status
**Get comprehensive portfolio analytics**

**Parameters:**
- `include_analytics` (boolean, optional, default: true): Include advanced analytics

**Example:**
```python
# Get full portfolio analysis
result = mcp__ai-news-trader__get_portfolio_status(
    include_analytics=True
)
```

**Expected Output:**
```json
{
    "portfolio_summary": {
        "total_value": 125432.67,
        "cash": 23456.78,
        "positions_value": 101975.89,
        "daily_pnl": 1234.56,
        "daily_return": 0.0099
    },
    "positions": [
        {
            "symbol": "AAPL",
            "quantity": 150,
            "current_price": 185.67,
            "position_value": 27850.50,
            "cost_basis": 183.45,
            "unrealized_pnl": 333.00,
            "percent_portfolio": 0.222
        }
    ],
    "analytics": {
        "sharpe_ratio": 1.87,
        "beta": 0.95,
        "alpha": 0.023,
        "max_drawdown": -0.056,
        "var_95": -2345.67
    }
}
```

**Best Practices:**
- Review daily before trading
- Monitor concentration risk
- Track against benchmarks
- Rebalance regularly

---

## Advanced Analytics Tools (7 tools)

### 11. run_backtest
**Comprehensive historical strategy testing**

**Parameters:**
- `strategy` (string, required): Strategy name to test
- `symbol` (string, required): Stock ticker symbol
- `start_date` (string, required): Start date (YYYY-MM-DD)
- `end_date` (string, required): End date
- `benchmark` (string, optional, default: "sp500"): Benchmark index
- `include_costs` (boolean, optional, default: true): Include transaction costs
- `use_gpu` (boolean, optional, default: true): Enable GPU acceleration

**Example:**
```python
# Backtest swing trading strategy
result = mcp__ai-news-trader__run_backtest(
    strategy="swing_trader",
    symbol="QQQ",
    start_date="2023-01-01",
    end_date="2024-12-31",
    benchmark="nasdaq",
    include_costs=True,
    use_gpu=True
)
```

**Expected Output:**
```json
{
    "backtest_summary": {
        "total_return": 0.342,
        "annualized_return": 0.168,
        "sharpe_ratio": 1.93,
        "sortino_ratio": 2.45,
        "max_drawdown": -0.124,
        "calmar_ratio": 1.35
    },
    "trade_statistics": {
        "total_trades": 156,
        "win_rate": 0.583,
        "profit_factor": 2.14,
        "avg_win": 0.0234,
        "avg_loss": -0.0109,
        "largest_win": 0.0876,
        "largest_loss": -0.0423
    },
    "monthly_returns": {
        "2023": [0.034, 0.021, -0.012, ...],
        "2024": [0.045, 0.018, 0.027, ...]
    }
}
```

**Best Practices:**
- Test across market cycles
- Include realistic costs
- Validate with out-of-sample data
- Compare multiple strategies

### 12. optimize_strategy
**Optimize strategy parameters using AI**

**Parameters:**
- `strategy` (string, required): Strategy to optimize
- `symbol` (string, required): Stock ticker symbol
- `parameter_ranges` (object, required): Parameters to optimize
- `optimization_metric` (string, optional, default: "sharpe_ratio"): Target metric
- `max_iterations` (integer, optional, default: 1000): Maximum iterations
- `use_gpu` (boolean, optional, default: true): Enable GPU acceleration

**Example:**
```python
# Optimize momentum strategy
result = mcp__ai-news-trader__optimize_strategy(
    strategy="momentum",
    symbol="SPY",
    parameter_ranges={
        "lookback_period": [10, 50],
        "entry_threshold": [0.01, 0.05],
        "stop_loss": [0.02, 0.10],
        "take_profit": [0.05, 0.20]
    },
    optimization_metric="sharpe_ratio",
    max_iterations=2000,
    use_gpu=True
)
```

**Expected Output:**
```json
{
    "optimization_results": {
        "optimal_parameters": {
            "lookback_period": 21,
            "entry_threshold": 0.023,
            "stop_loss": 0.045,
            "take_profit": 0.087
        },
        "performance_improvement": {
            "original_sharpe": 1.45,
            "optimized_sharpe": 2.23,
            "improvement": 0.537
        },
        "convergence_info": {
            "iterations": 1432,
            "convergence_time": 234.5,
            "gpu_speedup": "45.6x"
        }
    }
}
```

**Best Practices:**
- Use walk-forward optimization
- Avoid overfitting
- Validate on multiple assets
- Document parameter rationale

### 13. performance_report
**Generate detailed performance analytics**

**Parameters:**
- `strategy` (string, required): Strategy name
- `period_days` (integer, optional, default: 30): Analysis period
- `include_benchmark` (boolean, optional, default: true): Compare to benchmark
- `use_gpu` (boolean, optional, default: false): Enable GPU acceleration

**Example:**
```python
# Generate monthly performance report
result = mcp__ai-news-trader__performance_report(
    strategy="neural_swing",
    period_days=30,
    include_benchmark=True
)
```

**Expected Output:**
```json
{
    "performance_summary": {
        "period": "2024-12-11 to 2025-01-10",
        "total_return": 0.0456,
        "annualized_return": 0.587,
        "volatility": 0.156,
        "sharpe_ratio": 2.34,
        "information_ratio": 1.89
    },
    "risk_metrics": {
        "var_95": -0.0234,
        "cvar_95": -0.0345,
        "max_drawdown": -0.0567,
        "downside_deviation": 0.089
    },
    "attribution": {
        "alpha": 0.0123,
        "beta": 0.87,
        "market_timing": 0.0045,
        "stock_selection": 0.0078
    },
    "recommendations": [
        "Strategy outperforming benchmark by 18.5%",
        "Consider increasing position sizes",
        "Risk metrics within acceptable range"
    ]
}
```

**Best Practices:**
- Generate reports regularly
- Share with stakeholders
- Track metric trends
- Document insights

### 14. correlation_analysis
**Analyze multi-asset correlations**

**Parameters:**
- `symbols` (array, required): List of symbols to analyze
- `period_days` (integer, optional, default: 90): Analysis period
- `use_gpu` (boolean, optional, default: true): Enable GPU acceleration

**Example:**
```python
# Analyze tech stock correlations
result = mcp__ai-news-trader__correlation_analysis(
    symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
    period_days=180,
    use_gpu=True
)
```

**Expected Output:**
```json
{
    "correlation_matrix": {
        "AAPL": {"AAPL": 1.0, "MSFT": 0.78, "GOOGL": 0.72, "AMZN": 0.65, "NVDA": 0.81},
        "MSFT": {"AAPL": 0.78, "MSFT": 1.0, "GOOGL": 0.83, "AMZN": 0.69, "NVDA": 0.75},
        ...
    },
    "analysis": {
        "highest_correlation": {"pair": ["MSFT", "GOOGL"], "value": 0.83},
        "lowest_correlation": {"pair": ["AAPL", "AMZN"], "value": 0.65},
        "average_correlation": 0.74,
        "cluster_analysis": {
            "cluster_1": ["AAPL", "MSFT", "GOOGL"],
            "cluster_2": ["AMZN", "NVDA"]
        }
    },
    "diversification_score": 0.67
}
```

**Best Practices:**
- Update correlations regularly
- Use for portfolio construction
- Monitor correlation changes
- Consider rolling windows

### 15. run_benchmark
**Benchmark strategy and system performance**

**Parameters:**
- `strategy` (string, required): Strategy to benchmark
- `benchmark_type` (string, optional, default: "performance"): Type of benchmark
- `use_gpu` (boolean, optional, default: true): Enable GPU acceleration

**Example:**
```python
# Run comprehensive benchmark
result = mcp__ai-news-trader__run_benchmark(
    strategy="all",
    benchmark_type="comprehensive",
    use_gpu=True
)
```

**Expected Output:**
```json
{
    "benchmark_results": {
        "performance_metrics": {
            "trades_per_second": 15678,
            "latency_p99_ms": 2.3,
            "gpu_utilization": 0.76,
            "memory_usage_gb": 4.5
        },
        "strategy_rankings": [
            {"strategy": "neural_momentum", "sharpe": 2.45, "return": 0.324},
            {"strategy": "mean_reversion", "sharpe": 2.12, "return": 0.287},
            {"strategy": "swing_trader", "sharpe": 1.89, "return": 0.265}
        ],
        "system_capabilities": {
            "max_concurrent_strategies": 12,
            "neural_inference_speed": "8.3ms",
            "backtest_years_per_minute": 45.6
        }
    }
}
```

**Best Practices:**
- Run benchmarks before deployment
- Compare against baselines
- Monitor system resources
- Document performance targets

### 16. risk_analysis
**Comprehensive portfolio risk assessment**

**Parameters:**
- `portfolio` (array, required): Portfolio positions
- `var_confidence` (number, optional, default: 0.05): VaR confidence level
- `time_horizon` (integer, optional, default: 1): Time horizon in days
- `use_monte_carlo` (boolean, optional, default: true): Use Monte Carlo simulation
- `use_gpu` (boolean, optional, default: true): Enable GPU acceleration

**Example:**
```python
# Analyze portfolio risk
result = mcp__ai-news-trader__risk_analysis(
    portfolio=[
        {"symbol": "SPY", "quantity": 100, "value": 45678},
        {"symbol": "QQQ", "quantity": 50, "value": 23456},
        {"symbol": "TLT", "quantity": 200, "value": 18765}
    ],
    var_confidence=0.01,
    time_horizon=5,
    use_monte_carlo=True,
    use_gpu=True
)
```

**Expected Output:**
```json
{
    "risk_summary": {
        "portfolio_value": 87899,
        "var_99": -3456.78,
        "cvar_99": -4567.89,
        "expected_shortfall": -5234.56,
        "probability_of_loss": 0.423
    },
    "risk_decomposition": {
        "market_risk": 0.67,
        "specific_risk": 0.33,
        "concentration_risk": 0.23,
        "correlation_benefit": -0.12
    },
    "stress_tests": {
        "market_crash": {"loss": -12345.67, "probability": 0.05},
        "rate_spike": {"loss": -5678.90, "probability": 0.10},
        "volatility_surge": {"loss": -7890.12, "probability": 0.08}
    },
    "recommendations": [
        "Consider adding hedge positions",
        "Reduce concentration in technology",
        "Portfolio beta within target range"
    ]
}
```

**Best Practices:**
- Run daily risk assessments
- Stress test regularly
- Monitor risk limits
- Hedge tail risks

### 17. list_strategies
**List all available trading strategies**

**Parameters:** None

**Example:**
```python
# Get all strategies
result = mcp__ai-news-trader__list_strategies()
```

**Expected Output:**
```json
{
    "strategies": [
        {
            "name": "neural_momentum",
            "type": "trend_following",
            "description": "AI-enhanced momentum strategy with adaptive parameters",
            "performance": {"sharpe": 2.45, "annual_return": 0.324},
            "suitable_for": ["stocks", "etfs", "futures"]
        },
        {
            "name": "mean_reversion",
            "type": "contrarian",
            "description": "Statistical arbitrage using neural predictions",
            "performance": {"sharpe": 2.12, "annual_return": 0.287},
            "suitable_for": ["stocks", "pairs"]
        },
        ...
    ],
    "total_strategies": 12,
    "categories": ["trend_following", "mean_reversion", "arbitrage", "market_making"]
}
```

**Best Practices:**
- Review strategy documentation
- Match strategies to market conditions
- Diversify across strategies
- Monitor strategy correlations

---

## News & Sentiment Tools (2 tools)

### 18. analyze_news
**AI-powered news sentiment analysis**

**Parameters:**
- `symbol` (string, required): Stock ticker symbol
- `lookback_hours` (integer, optional, default: 24): Hours to look back
- `sentiment_model` (string, optional, default: "enhanced"): Model to use
- `use_gpu` (boolean, optional, default: false): Enable GPU acceleration

**Example:**
```python
# Analyze Tesla news sentiment
result = mcp__ai-news-trader__analyze_news(
    symbol="TSLA",
    lookback_hours=48,
    sentiment_model="enhanced",
    use_gpu=True
)
```

**Expected Output:**
```json
{
    "sentiment_analysis": {
        "overall_sentiment": 0.67,
        "sentiment_label": "bullish",
        "confidence": 0.89,
        "news_volume": 145,
        "trending_score": 0.92
    },
    "key_topics": [
        {"topic": "earnings_beat", "sentiment": 0.85, "mentions": 23},
        {"topic": "new_product", "sentiment": 0.72, "mentions": 18},
        {"topic": "competition", "sentiment": -0.34, "mentions": 12}
    ],
    "influential_articles": [
        {
            "title": "Tesla Beats Q4 Earnings Expectations",
            "source": "Reuters",
            "sentiment": 0.89,
            "impact_score": 0.95
        }
    ],
    "trading_signal": {
        "recommendation": "buy",
        "strength": 0.73,
        "time_horizon": "short_term"
    }
}
```

**Best Practices:**
- Combine with price action
- Monitor sentiment shifts
- Track news volume spikes
- Validate with fundamentals

### 19. get_news_sentiment
**Real-time news sentiment data**

**Parameters:**
- `symbol` (string, required): Stock ticker symbol
- `sources` (array, optional): Specific news sources

**Example:**
```python
# Get Apple sentiment from major sources
result = mcp__ai-news-trader__get_news_sentiment(
    symbol="AAPL",
    sources=["reuters", "bloomberg", "wsj", "cnbc"]
)
```

**Expected Output:**
```json
{
    "real_time_sentiment": {
        "symbol": "AAPL",
        "current_sentiment": 0.54,
        "sentiment_trend": "improving",
        "last_update": "2025-01-10T15:45:00Z"
    },
    "source_breakdown": {
        "reuters": {"sentiment": 0.67, "articles": 12},
        "bloomberg": {"sentiment": 0.48, "articles": 8},
        "wsj": {"sentiment": 0.52, "articles": 6},
        "cnbc": {"sentiment": 0.49, "articles": 15}
    },
    "intraday_sentiment": [
        {"time": "09:30", "sentiment": 0.45},
        {"time": "10:30", "sentiment": 0.48},
        {"time": "11:30", "sentiment": 0.52},
        {"time": "12:30", "sentiment": 0.54}
    ]
}
```

**Best Practices:**
- Track sentiment momentum
- Compare across sources
- Use for timing entries
- Set sentiment alerts

---

## System Tools (2 tools)

### 20. ping
**Test server connectivity**

**Parameters:** None

**Example:**
```python
# Check MCP server status
result = mcp__ai-news-trader__ping()
```

**Expected Output:**
```json
{
    "status": "online",
    "server_time": "2025-01-10T15:45:30.123Z",
    "version": "2.1.0",
    "gpu_available": true,
    "active_connections": 3,
    "uptime_hours": 168.5
}
```

**Best Practices:**
- Check before trading sessions
- Monitor during operations
- Verify after updates
- Test from different locations

### 21. get_strategy_info
**Get detailed strategy information**

**Parameters:**
- `strategy` (string, required): Strategy name

**Example:**
```python
# Get neural momentum details
result = mcp__ai-news-trader__get_strategy_info(
    strategy="neural_momentum"
)
```

**Expected Output:**
```json
{
    "strategy_details": {
        "name": "neural_momentum",
        "version": "3.2.1",
        "description": "Advanced momentum strategy using NHITS neural forecasting",
        "parameters": {
            "lookback_period": 20,
            "forecast_horizon": 24,
            "entry_threshold": 0.023,
            "stop_loss": 0.045,
            "take_profit": 0.087
        },
        "requirements": {
            "min_data_points": 5000,
            "gpu_recommended": true,
            "update_frequency": "1min"
        },
        "performance_history": {
            "last_30_days": {"return": 0.045, "sharpe": 2.34},
            "last_90_days": {"return": 0.132, "sharpe": 2.12},
            "ytd": {"return": 0.287, "sharpe": 2.45}
        }
    }
}
```

**Best Practices:**
- Review before deployment
- Check version compatibility
- Understand parameters
- Monitor performance trends

---

## Common Trading Workflows

### 1. Neural-Enhanced Day Trading
```python
# Morning analysis
analysis = mcp__ai-news-trader__quick_analysis("SPY", use_gpu=True)
sentiment = mcp__ai-news-trader__analyze_news("SPY", lookback_hours=12)
forecast = mcp__ai-news-trader__neural_forecast("SPY", horizon=8, use_gpu=True)

# If bullish signals align
if forecast["forecast"]["point_forecast"][0] > analysis["current_price"] and sentiment["overall_sentiment"] > 0.6:
    # Simulate first
    sim = mcp__ai-news-trader__simulate_trade("neural_momentum", "SPY", "buy")
    
    # Execute if simulation positive
    if sim["expected_outcomes"]["4h"]["pnl"] > 0:
        trade = mcp__ai-news_trader__execute_trade(
            strategy="neural_momentum",
            symbol="SPY",
            action="buy",
            quantity=100,
            order_type="limit",
            limit_price=analysis["current_price"] * 0.999
        )
```

### 2. Portfolio Risk Management
```python
# Get current positions
portfolio = mcp__ai-news-trader__get_portfolio_status(include_analytics=True)

# Analyze risk
risk = mcp__ai-news-trader__risk_analysis(
    portfolio=portfolio["positions"],
    var_confidence=0.01,
    time_horizon=5,
    use_gpu=True
)

# Check correlations for diversification
symbols = [pos["symbol"] for pos in portfolio["positions"]]
correlations = mcp__ai-news-trader__correlation_analysis(symbols, period_days=90)

# Rebalance if needed
if risk["risk_summary"]["concentration_risk"] > 0.3:
    # Execute rebalancing trades
    pass
```

### 3. Strategy Development & Optimization
```python
# Backtest initial strategy
backtest = mcp__ai-news-trader__run_backtest(
    strategy="mean_reversion",
    symbol="QQQ",
    start_date="2023-01-01",
    end_date="2024-12-31",
    use_gpu=True
)

# Optimize if underperforming
if backtest["backtest_summary"]["sharpe_ratio"] < 1.5:
    optimization = mcp__ai-news-trader__optimize_strategy(
        strategy="mean_reversion",
        symbol="QQQ",
        parameter_ranges={
            "lookback": [10, 50],
            "z_score_entry": [1.5, 3.0],
            "z_score_exit": [0.0, 1.0]
        },
        optimization_metric="sharpe_ratio",
        use_gpu=True
    )
    
    # Deploy optimized parameters
    optimal_params = optimization["optimization_results"]["optimal_parameters"]
```

### 4. Neural Model Training Pipeline
```python
# Train new model
training = mcp__ai-news-trader__neural_train(
    data_path="/data/spy_5min_2years.csv",
    model_type="nhits",
    epochs=300,
    use_gpu=True
)

# Evaluate performance
evaluation = mcp__ai-news-trader__neural_evaluate(
    model_id=training["model_id"],
    test_data="/data/spy_5min_recent.csv",
    metrics=["mae", "directional_accuracy"]
)

# Backtest if evaluation passes
if evaluation["evaluation_metrics"]["directional_accuracy"] > 0.6:
    neural_backtest = mcp__ai-news-trader__neural_backtest(
        model_id=training["model_id"],
        start_date="2024-01-01",
        end_date="2024-12-31"
    )
    
    # Deploy if profitable
    if neural_backtest["backtest_results"]["sharpe_ratio"] > 2.0:
        # Model ready for production
        pass
```

### 5. Real-Time Trading Operations
```python
# Pre-market setup
ping = mcp__ai-news-trader__ping()
strategies = mcp__ai-news-trader__list_strategies()
portfolio = mcp__ai-news-trader__get_portfolio_status()

# Market hours loop
while market_open:
    # Get signals for watchlist
    for symbol in watchlist:
        # Neural forecast
        forecast = mcp__ai-news-trader__neural_forecast(symbol, horizon=4)
        
        # News check
        news = mcp__ai-news_trader__get_news_sentiment(symbol)
        
        # Technical analysis
        analysis = mcp__ai-news-trader__quick_analysis(symbol)
        
        # Trade if conditions met
        if should_trade(forecast, news, analysis):
            execute_trading_logic(symbol)
    
    # Risk monitoring
    risk_check = mcp__ai-news-trader__risk_analysis(get_current_positions())
    
    time.sleep(60)  # Check every minute
```

---

## Best Practices Summary

### Performance Optimization
1. **Always use GPU acceleration** for neural operations (`use_gpu=True`)
2. **Batch operations** when analyzing multiple symbols
3. **Cache frequently used data** to reduce API calls
4. **Use appropriate time horizons** for your trading style

### Risk Management
1. **Run risk analysis daily** before trading
2. **Set position size limits** based on portfolio risk
3. **Use stop-loss orders** on all positions
4. **Monitor correlation changes** for diversification

### Model Management
1. **Retrain models monthly** with recent data
2. **Track model drift** with regular evaluations
3. **A/B test new models** before full deployment
4. **Document all model versions** and parameters

### News & Sentiment
1. **Combine sentiment with price action** for confirmation
2. **Track sentiment momentum** not just absolute values
3. **Use multiple news sources** to avoid bias
4. **Set alerts for sentiment shifts** on key positions

### System Operations
1. **Check server status** before trading sessions
2. **Monitor GPU utilization** during peak loads
3. **Backup model checkpoints** regularly
4. **Log all trades** for audit trail

---

## Error Handling

All tools return structured error responses:

```json
{
    "error": {
        "code": "INVALID_SYMBOL",
        "message": "Symbol XYZ not found",
        "details": "Please check the symbol and try again",
        "timestamp": "2025-01-10T15:45:30.123Z"
    }
}
```

Common error codes:
- `INVALID_SYMBOL`: Symbol not found
- `INSUFFICIENT_DATA`: Not enough historical data
- `MODEL_NOT_FOUND`: Specified model doesn't exist
- `GPU_UNAVAILABLE`: GPU requested but not available
- `RATE_LIMIT`: API rate limit exceeded
- `INVALID_PARAMETERS`: Parameter validation failed

---

## Support & Resources

- **Documentation**: `/workspaces/ai-news-trader/.claude/commands/`
- **Examples**: `examples/mcp_trading_examples.py`
- **Models**: `models/` directory for trained models
- **Logs**: `logs/mcp_server.log` for debugging

For issues or questions, check the error logs and ensure all parameters are correctly specified.