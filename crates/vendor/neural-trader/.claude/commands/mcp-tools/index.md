# MCP Tools Index - AI News Trading Platform

## Overview
The AI News Trading Platform provides 21 advanced tools via Model Context Protocol (MCP) for neural forecasting, trading strategy development, and market analysis. All tools are accessible through the MCP server (`python src/mcp/mcp_server_enhanced.py`).

## Quick Reference Table

| Tool Name | Category | GPU Support | Description |
|-----------|----------|-------------|-------------|
| `ping` | System | No | Verify server connectivity |
| `list_strategies` | System | No | List all available trading strategies |
| `get_strategy_info` | Trading Strategy | No | Get detailed strategy information |
| `quick_analysis` | Trading Strategy | Optional | Real-time market analysis |
| `simulate_trade` | Trading Strategy | Optional | Trade simulation with tracking |
| `execute_trade` | Trading Strategy | No | Live trade execution (demo mode) |
| `get_portfolio_status` | Trading Strategy | No | Portfolio analytics and metrics |
| `analyze_news` | News & Sentiment | Optional | AI-powered news sentiment analysis |
| `get_news_sentiment` | News & Sentiment | No | Real-time sentiment data |
| `run_backtest` | Analytics | Yes | Historical strategy testing |
| `optimize_strategy` | Analytics | Yes | Parameter optimization |
| `performance_report` | Analytics | Optional | Detailed performance analytics |
| `correlation_analysis` | Analytics | Yes | Multi-asset correlation analysis |
| `risk_analysis` | Analytics | Yes | Portfolio risk assessment |
| `run_benchmark` | Analytics | Yes | System performance benchmarks |
| `neural_forecast` | Neural | Yes | AI price predictions |
| `neural_train` | Neural | Yes | Train forecasting models |
| `neural_evaluate` | Neural | Yes | Evaluate model performance |
| `neural_backtest` | Neural | Yes | Historical model validation |
| `neural_model_status` | Neural | No | Monitor model health |
| `neural_optimize` | Neural | Yes | Hyperparameter optimization |

## Tool Categories

### 🔧 System Tools (2 tools)
Essential tools for system connectivity and configuration.

#### [`ping`](ping.md)
- **Purpose**: Verify MCP server connectivity
- **Usage**: Health checks and debugging
```bash
mcp call ping
```

#### [`list_strategies`](list-strategies.md)
- **Purpose**: List all available trading strategies with GPU capabilities
- **Usage**: Discover available strategies before trading
```bash
mcp call list_strategies
```

### 📈 Trading Strategy Tools (4 tools)
Core trading operations and portfolio management.

#### [`get_strategy_info`](get-strategy-info.md)
- **Purpose**: Get detailed information about a specific trading strategy
- **Parameters**: strategy name
```bash
mcp call get_strategy_info '{"strategy": "momentum"}'
```

#### [`quick_analysis`](quick-analysis.md)
- **Purpose**: Get quick market analysis with optional GPU acceleration
- **Parameters**: symbol, use_gpu
- **Best for**: Real-time decision making
```bash
mcp call quick_analysis '{"symbol": "AAPL", "use_gpu": true}'
```

#### [`simulate_trade`](simulate-trade.md)
- **Purpose**: Simulate trading operations with performance tracking
- **Parameters**: strategy, symbol, action, use_gpu
- **Best for**: Testing strategies without risk
```bash
mcp call simulate_trade '{"strategy": "momentum", "symbol": "TSLA", "action": "buy", "use_gpu": true}'
```

#### [`get_portfolio_status`](get-portfolio-status.md)
- **Purpose**: Get current portfolio status with advanced analytics
- **Parameters**: include_analytics
- **Best for**: Portfolio monitoring
```bash
mcp call get_portfolio_status '{"include_analytics": true}'
```

### 📰 News & Sentiment Tools (2 tools)
AI-powered news analysis for market sentiment.

#### [`analyze_news`](analyze-news.md) ⭐
- **Purpose**: Deep AI sentiment analysis of market news
- **Parameters**: symbol, lookback_hours, sentiment_model, use_gpu
- **Best for**: Comprehensive sentiment analysis
```bash
mcp call analyze_news '{"symbol": "AAPL", "lookback_hours": 24, "sentiment_model": "enhanced", "use_gpu": true}'
```

#### [`get_news_sentiment`](get-news-sentiment.md) ⚡
- **Purpose**: Real-time news sentiment data
- **Parameters**: symbol, sources
- **Best for**: Fast sentiment checks (<100ms)
```bash
mcp call get_news_sentiment '{"symbol": "AAPL", "sources": ["reuters", "bloomberg"]}'
```

### 📊 Advanced Analytics Tools (7 tools)
Comprehensive analysis and optimization capabilities.

#### [`run_backtest`](run-backtest.md)
- **Purpose**: Run historical backtests with GPU acceleration
- **Parameters**: strategy, symbol, start_date, end_date, benchmark, include_costs, use_gpu
```bash
mcp call run_backtest '{"strategy": "momentum", "symbol": "AAPL", "start_date": "2024-01-01", "end_date": "2024-12-31", "use_gpu": true}'
```

#### [`optimize_strategy`](optimize-strategy.md)
- **Purpose**: Optimize strategy parameters using GPU
- **Parameters**: strategy, symbol, parameter_ranges, optimization_metric, max_iterations, use_gpu
```bash
mcp call optimize_strategy '{"strategy": "momentum", "symbol": "AAPL", "parameter_ranges": {"window": [10, 50], "threshold": [0.01, 0.05]}, "use_gpu": true}'
```

#### [`performance_report`](performance-report.md)
- **Purpose**: Generate detailed performance analytics
- **Parameters**: strategy, period_days, include_benchmark, use_gpu
```bash
mcp call performance_report '{"strategy": "momentum", "period_days": 30, "include_benchmark": true}'
```

#### [`correlation_analysis`](correlation-analysis.md)
- **Purpose**: Analyze correlations between multiple assets
- **Parameters**: symbols, period_days, use_gpu
```bash
mcp call correlation_analysis '{"symbols": ["AAPL", "MSFT", "GOOGL"], "period_days": 90, "use_gpu": true}'
```

#### [`risk_analysis`](risk-analysis.md)
- **Purpose**: Comprehensive portfolio risk assessment
- **Parameters**: portfolio, time_horizon, var_confidence, use_monte_carlo, use_gpu
```bash
mcp call risk_analysis '{"portfolio": [{"symbol": "AAPL", "weight": 0.5}, {"symbol": "BONDS", "weight": 0.5}], "use_gpu": true}'
```

#### [`execute_trade`](execute-trade.md)
- **Purpose**: Execute live trades with order management (demo mode)
- **Parameters**: strategy, symbol, action, quantity, order_type, limit_price
```bash
mcp call execute_trade '{"strategy": "momentum", "symbol": "AAPL", "action": "buy", "quantity": 100}'
```

#### [`run_benchmark`](run-benchmark.md)
- **Purpose**: Benchmark strategy and system performance
- **Parameters**: strategy, benchmark_type, use_gpu
```bash
mcp call run_benchmark '{"strategy": "all", "benchmark_type": "performance", "use_gpu": true}'
```

### 🧠 Neural Forecasting Tools (6 tools)
Advanced AI-powered price prediction and model management.

#### [`neural_forecast`](neural-forecast.md) ⭐
- **Purpose**: Generate AI price predictions with confidence intervals
- **Parameters**: symbol, horizon, confidence_level, model_id, use_gpu
- **Best for**: Price predictions up to 72 hours
```bash
mcp call neural_forecast '{"symbol": "AAPL", "horizon": 24, "confidence_level": 0.95, "use_gpu": true}'
```

#### [`neural_train`](neural-train.md)
- **Purpose**: Train custom neural forecasting models
- **Parameters**: data_path, model_type, epochs, batch_size, learning_rate, validation_split, use_gpu
```bash
mcp call neural_train '{"data_path": "data/training.csv", "model_type": "nhits", "epochs": 200, "use_gpu": true}'
```

#### [`neural_evaluate`](neural-evaluate.md)
- **Purpose**: Evaluate trained models on test data
- **Parameters**: model_id, test_data, metrics, use_gpu
```bash
mcp call neural_evaluate '{"model_id": "model_001", "test_data": "data/test.csv", "metrics": ["mae", "rmse"], "use_gpu": true}'
```

#### [`neural_backtest`](neural-backtest.md)
- **Purpose**: Backtest neural models against benchmarks
- **Parameters**: model_id, start_date, end_date, benchmark, rebalance_frequency, use_gpu
```bash
mcp call neural_backtest '{"model_id": "model_001", "start_date": "2024-01-01", "end_date": "2024-12-31", "use_gpu": true}'
```

#### [`neural_model_status`](neural-model-status.md)
- **Purpose**: Get status of neural models
- **Parameters**: model_id (optional)
```bash
mcp call neural_model_status
```

#### [`neural_optimize`](neural-optimize.md)
- **Purpose**: Optimize neural model hyperparameters
- **Parameters**: model_id, parameter_ranges, optimization_metric, trials, use_gpu
```bash
mcp call neural_optimize '{"model_id": "model_001", "parameter_ranges": {"learning_rate": [0.0001, 0.01]}, "trials": 100, "use_gpu": true}'
```

## Common Workflow Combinations

### 1. News-Driven Trading Workflow
```bash
# Step 1: Check real-time sentiment
mcp call get_news_sentiment '{"symbol": "AAPL"}'

# Step 2: Deep sentiment analysis if significant news
mcp call analyze_news '{"symbol": "AAPL", "lookback_hours": 24, "use_gpu": true}'

# Step 3: Get neural forecast
mcp call neural_forecast '{"symbol": "AAPL", "horizon": 24, "use_gpu": true}'

# Step 4: Simulate trade based on combined signals
mcp call simulate_trade '{"strategy": "news_momentum", "symbol": "AAPL", "action": "buy", "use_gpu": true}'
```

### 2. Strategy Development Workflow
```bash
# Step 1: List available strategies
mcp call list_strategies

# Step 2: Get strategy details
mcp call get_strategy_info '{"strategy": "momentum"}'

# Step 3: Run backtest
mcp call run_backtest '{"strategy": "momentum", "symbol": "AAPL", "start_date": "2024-01-01", "end_date": "2024-12-31", "use_gpu": true}'

# Step 4: Optimize parameters
mcp call optimize_strategy '{"strategy": "momentum", "symbol": "AAPL", "parameter_ranges": {"window": [10, 50]}, "use_gpu": true}'

# Step 5: Generate performance report
mcp call performance_report '{"strategy": "momentum", "period_days": 30}'
```

### 3. Neural Model Development
```bash
# Step 1: Train model
mcp call neural_train '{"data_path": "data/AAPL_history.csv", "model_type": "nhits", "epochs": 200, "use_gpu": true}'

# Step 2: Evaluate performance
mcp call neural_evaluate '{"model_id": "nhits_AAPL", "test_data": "data/AAPL_test.csv", "use_gpu": true}'

# Step 3: Optimize hyperparameters
mcp call neural_optimize '{"model_id": "nhits_AAPL", "parameter_ranges": {"learning_rate": [0.0001, 0.01]}, "use_gpu": true}'

# Step 4: Backtest model
mcp call neural_backtest '{"model_id": "nhits_AAPL", "start_date": "2024-01-01", "end_date": "2024-12-31", "use_gpu": true}'
```

### 4. Portfolio Risk Management
```bash
# Step 1: Get portfolio status
mcp call get_portfolio_status '{"include_analytics": true}'

# Step 2: Analyze correlations
mcp call correlation_analysis '{"symbols": ["AAPL", "MSFT", "GOOGL", "AMZN"], "use_gpu": true}'

# Step 3: Risk assessment
mcp call risk_analysis '{"portfolio": [{"symbol": "AAPL", "weight": 0.3}, {"symbol": "MSFT", "weight": 0.3}, {"symbol": "BONDS", "weight": 0.4}], "use_gpu": true}'

# Step 4: Benchmark performance
mcp call run_benchmark '{"strategy": "diversified", "benchmark_type": "risk_adjusted", "use_gpu": true}'
```

## Best Practices Summary

### 1. GPU Acceleration
- **Always use GPU** for neural operations (training, forecasting, optimization)
- **Consider GPU** for heavy analytics (backtesting, risk analysis)
- **Skip GPU** for simple queries (status checks, real-time sentiment)

### 2. Tool Selection
- **Real-time needs**: Use `get_news_sentiment` over `analyze_news`
- **Deep analysis**: Use `analyze_news` with GPU for comprehensive insights
- **Price predictions**: Combine `neural_forecast` with sentiment analysis
- **Risk management**: Run `risk_analysis` before executing trades

### 3. Performance Tips
- **Batch operations** when analyzing multiple symbols
- **Cache results** for frequently accessed data (5-15 minute validity)
- **Use appropriate timeframes** - shorter for day trading, longer for investing
- **Monitor model health** with `neural_model_status` regularly

### 4. Integration Patterns
- **Combine tools** for stronger signals (sentiment + technical + neural)
- **Validate strategies** with backtesting before live trading
- **Optimize parameters** based on recent market conditions
- **Monitor continuously** with real-time tools during market hours

## Getting Started

1. **Start the MCP server**:
   ```bash
   python src/mcp/mcp_server_enhanced.py
   ```

2. **Verify connectivity**:
   ```bash
   mcp call ping
   ```

3. **List available strategies**:
   ```bash
   mcp call list_strategies
   ```

4. **Begin with analysis**:
   ```bash
   mcp call quick_analysis '{"symbol": "AAPL", "use_gpu": true}'
   ```

## Additional Resources
- [Neural Forecasting Commands](../neural-forecasting.md)
- [Trading Strategies Guide](../trading-strategies.md)
- [Memory System](../memory-system.md)
- [System Orchestration](../system-orchestration.md)
- [Quick Reference](../quick-reference.md)