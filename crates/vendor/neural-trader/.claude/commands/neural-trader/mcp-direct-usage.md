# Direct MCP Tool Usage in Claude Code

## Overview
You can call MCP tools directly in Claude Code using natural language or the direct syntax. The MCP server `ai-news-trader` is already configured and ready to use.

## Direct Tool Syntax
```
ai-news-trader:tool_name (MCP)(parameter: value, parameter2: value2)
```

## All 21 MCP Tools - Direct Usage Examples

### Neural Forecasting Tools

#### 1. neural_forecast
```
ai-news-trader:neural_forecast (MCP)(symbol: "AAPL", horizon: 24, confidence_level: 0.95, use_gpu: true)
```
Or simply ask: "Generate a 24-hour neural forecast for AAPL with 95% confidence using GPU"

#### 2. neural_train
```
ai-news-trader:neural_train (MCP)(data_path: "data/spy_5min.csv", model_type: "nhits", epochs: 200, use_gpu: true)
```
Or: "Train an NHITS neural model on spy_5min.csv for 200 epochs with GPU"

#### 3. neural_evaluate
```
ai-news-trader:neural_evaluate (MCP)(model_id: "nhits_v3", test_data: "data/test.csv", metrics: ["mae", "rmse"], use_gpu: true)
```
Or: "Evaluate the nhits_v3 model on test.csv and show MAE and RMSE metrics"

#### 4. neural_backtest
```
ai-news-trader:neural_backtest (MCP)(model_id: "transformer_v2", start_date: "2024-01-01", end_date: "2024-12-31", benchmark: "sp500")
```
Or: "Backtest transformer_v2 model for all of 2024 against S&P 500"

#### 5. neural_model_status
```
ai-news-trader:neural_model_status (MCP)(model_id: "nhits_v3")
```
Or: "Check the status of nhits_v3 model"

#### 6. neural_optimize
```
ai-news-trader:neural_optimize (MCP)(model_id: "nhits_v3", parameter_ranges: {"learning_rate": [0.0001, 0.01], "batch_size": [32, 256]}, trials: 100)
```
Or: "Optimize nhits_v3 hyperparameters for learning rate and batch size with 100 trials"

### Trading Strategy Tools

#### 7. quick_analysis
```
ai-news-trader:quick_analysis (MCP)(symbol: "TSLA", use_gpu: true)
```
Or: "Give me a quick market analysis for TSLA"

#### 8. simulate_trade
```
ai-news-trader:simulate_trade (MCP)(strategy: "momentum", symbol: "NVDA", action: "buy", use_gpu: true)
```
Or: "Simulate a momentum buy trade on NVDA"

#### 9. execute_trade
```
ai-news-trader:execute_trade (MCP)(strategy: "swing", symbol: "SPY", action: "buy", quantity: 100, order_type: "limit", limit_price: 450.50)
```
Or: "Execute a limit buy order for 100 shares of SPY at $450.50 using swing strategy"

#### 10. get_portfolio_status
```
ai-news-trader:get_portfolio_status (MCP)(include_analytics: true)
```
Or: "Show my current portfolio status with analytics"

### Advanced Analytics Tools

#### 11. run_backtest
```
ai-news-trader:run_backtest (MCP)(strategy: "momentum", symbol: "AAPL", start_date: "2024-01-01", end_date: "2024-06-01", use_gpu: true)
```
Or: "Backtest momentum strategy on AAPL from January to June 2024"

#### 12. optimize_strategy
```
ai-news-trader:optimize_strategy (MCP)(strategy: "swing", symbol: "MSFT", parameter_ranges: {"window": [10, 50], "threshold": [0.01, 0.05]}, use_gpu: true)
```
Or: "Optimize swing trading parameters for MSFT"

#### 13. performance_report
```
ai-news-trader:performance_report (MCP)(strategy: "momentum", period_days: 30, include_benchmark: true)
```
Or: "Generate 30-day performance report for momentum strategy with benchmark"

#### 14. correlation_analysis
```
ai-news-trader:correlation_analysis (MCP)(symbols: ["AAPL", "MSFT", "GOOGL", "AMZN"], period_days: 90, use_gpu: true)
```
Or: "Analyze correlations between AAPL, MSFT, GOOGL, and AMZN over 90 days"

#### 15. run_benchmark
```
ai-news-trader:run_benchmark (MCP)(strategy: "all", benchmark_type: "performance", use_gpu: true)
```
Or: "Run performance benchmarks for all strategies"

#### 16. risk_analysis
```
ai-news-trader:risk_analysis (MCP)(portfolio: [{"symbol": "AAPL", "weight": 0.4}, {"symbol": "GOOGL", "weight": 0.3}, {"symbol": "MSFT", "weight": 0.3}], var_confidence: 0.05, use_gpu: true)
```
Or: "Analyze risk for portfolio: 40% AAPL, 30% GOOGL, 30% MSFT with 95% VaR"

#### 17. list_strategies
```
ai-news-trader:list_strategies (MCP)()
```
Or: "List all available trading strategies"

### News & Sentiment Tools

#### 18. analyze_news
```
ai-news-trader:analyze_news (MCP)(symbol: "NVDA", lookback_hours: 24, sentiment_model: "enhanced", use_gpu: true)
```
Or: "Analyze NVDA news sentiment for the last 24 hours"

#### 19. get_news_sentiment
```
ai-news-trader:get_news_sentiment (MCP)(symbol: "TSLA", sources: ["reuters", "bloomberg"])
```
Or: "Get Tesla news sentiment from Reuters and Bloomberg"

### System Tools

#### 20. ping
```
ai-news-trader:ping (MCP)()
```
Or: "Ping the AI news trader server"

#### 21. get_strategy_info
```
ai-news-trader:get_strategy_info (MCP)(strategy: "momentum")
```
Or: "Get detailed info about the momentum strategy"

## Natural Language Examples

### Complex Workflow
"I want to analyze AAPL: first check the news sentiment, then run a neural forecast for 48 hours, and finally do a quick technical analysis. Use GPU for everything."

Claude Code will execute:
1. `ai-news-trader:analyze_news` for AAPL
2. `ai-news-trader:neural_forecast` with 48-hour horizon
3. `ai-news-trader:quick_analysis` for technical indicators

### Parallel Analysis
"Analyze AAPL, MSFT, and GOOGL simultaneously - I need quick analysis and 24-hour forecasts for each"

Claude Code will run 6 parallel MCP calls:
- 3 `quick_analysis` calls
- 3 `neural_forecast` calls

### Conditional Trading
"If NVDA sentiment is above 80% positive, simulate a momentum trade"

Claude Code will:
1. Check sentiment with `analyze_news`
2. If condition met, call `simulate_trade`

## Tips for Direct Usage

1. **Case Insensitive**: Tool names and parameters work in any case
2. **Flexible Syntax**: Natural language often works better than direct syntax
3. **Parallel Execution**: Ask for multiple analyses "simultaneously" or "in parallel"
4. **Chaining**: Use "then" or "after that" to chain operations
5. **Memory Storage**: Add "and store the results as [name]" to save to memory

## Common Patterns

### Morning Analysis
"Good morning! Please run my daily analysis: check portfolio status, analyze news for my top 5 holdings, and generate 24-hour forecasts for each"

### Risk Check
"Before I trade, run a comprehensive risk analysis on my current portfolio and suggest any necessary adjustments"

### Strategy Optimization
"Optimize my swing trading strategy for the current market conditions using the last 3 months of data"

### Real-time Monitoring
"Monitor AAPL, TSLA, and SPY - alert me if any show >85% bullish sentiment or neural forecast predicts >3% move"

## Error Handling

If a tool fails, Claude Code will:
1. Explain the error
2. Suggest alternatives
3. Offer to retry with different parameters

Example: "The neural forecast failed due to insufficient data. Would you like me to try a shorter horizon or use technical analysis instead?"

## Quick Reference Card

```
NEWS & SENTIMENT
- Analyze news: "Check AAPL news sentiment"
- Get sentiment: "Show me Tesla sentiment scores"

NEURAL FORECASTING  
- Forecast: "Predict NVDA price for next 24 hours"
- Train: "Train new model on my data"
- Evaluate: "Test model accuracy"

TRADING
- Analysis: "Quick analysis of SPY"
- Simulate: "Simulate momentum trade on GOOGL"
- Portfolio: "Show my portfolio status"

ANALYTICS
- Backtest: "Test strategy on historical data"
- Optimize: "Find best parameters for swing trading"
- Risk: "Analyze portfolio risk"

SYSTEM
- Ping: "Check if server is running"
- Strategies: "List all strategies"
```

Remember: You don't need to memorize the exact syntax - just describe what you want to do, and Claude Code will handle the MCP tool calls for you!