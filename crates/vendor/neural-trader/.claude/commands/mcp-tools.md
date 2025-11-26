# MCP Tools Reference - Native Claude Code Integration

> **📚 New Detailed Documentation Available!**  
> For comprehensive documentation of individual MCP tools, see the new [`mcp-tools/`](./mcp-tools/) directory:
> - [`mcp-tools/index.md`](./mcp-tools/index.md) - Complete index and quick reference for all 21 tools
> - [`mcp-tools/analyze-news.md`](./mcp-tools/analyze-news.md) - Detailed guide for AI news analysis
> - [`mcp-tools/get-news-sentiment.md`](./mcp-tools/get-news-sentiment.md) - Real-time sentiment monitoring
> - Additional tool-specific documentation coming soon!

## Overview
The Enhanced MCP Server provides 21 advanced trading tools directly accessible through Claude Code's native MCP integration. These tools offer neural forecasting, trading strategy execution, market analysis, and portfolio management capabilities.

## Native MCP Tool Access
All tools are available directly in Claude Code using the prefix `mcp__ai-news-trader__`. No command-line execution needed - just ask Claude to use these tools naturally.

### Example Usage:
```
"Use the neural_forecast tool to predict AAPL prices for the next 24 hours"
"Analyze news sentiment for TSLA using the analyze_news tool"
"Get my current portfolio status with analytics"
```

## Neural Forecasting Tools (6 tools)

### mcp__ai-news-trader__neural_forecast
Generate neural network forecasts with confidence intervals.

**Parameters:**
- `symbol` (string): Trading symbol (e.g., "AAPL")
- `horizon` (int): Forecast horizon in hours
- `confidence_level` (float): Confidence level (default: 0.95)
- `use_gpu` (bool): Enable GPU acceleration
- `model_id` (string, optional): Specific model to use

**Native Claude Code Example:**
```
"Use neural_forecast to predict AAPL prices for the next 24 hours with GPU acceleration"
```

**Direct Tool Call:**
```json
{
  "tool": "mcp__ai-news-trader__neural_forecast",
  "parameters": {
    "symbol": "AAPL",
    "horizon": 24,
    "confidence_level": 0.95,
    "use_gpu": true
  }
}
```

**Expected Return:**
```json
{
  "symbol": "AAPL",
  "current_price": 185.50,
  "forecast": {
    "24h": 187.25,
    "confidence_interval": [185.75, 188.75],
    "trend": "bullish",
    "model_accuracy": 0.92
  },
  "gpu_time_ms": 8.3
}
```

### mcp__ai-news-trader__neural_train
Train neural forecasting models on custom datasets.

**Parameters:**
- `data_path` (string): Path to training dataset
- `model_type` (string): Model type (LSTM, Transformer, GRU, CNN_LSTM)
- `epochs` (int): Training epochs (default: 100)
- `validation_split` (float): Validation split (default: 0.2)
- `batch_size` (int): Batch size (default: 32)
- `learning_rate` (float): Learning rate (default: 0.001)
- `use_gpu` (bool): Enable GPU acceleration

**Native Claude Code Example:**
```
"Train a Transformer neural model on trading_data.csv with 200 epochs using GPU"
```

**Direct Tool Call:**
```json
{
  "tool": "mcp__ai-news-trader__neural_train",
  "parameters": {
    "data_path": "trading_data.csv",
    "model_type": "Transformer",
    "epochs": 200,
    "use_gpu": true
  }
}
```

### mcp__ai-news-trader__neural_evaluate
Evaluate trained neural models with comprehensive metrics.

**Parameters:**
- `model_id` (string): Model identifier
- `test_data` (string): Path to test dataset
- `metrics` (list): Evaluation metrics ["mae", "rmse", "mape", "r2_score"]
- `use_gpu` (bool): Enable GPU acceleration

**Native Claude Code Example:**
```
"Evaluate the transformer_model_001 on test_data.csv with all metrics"
```

### neural_backtest
Run historical backtesting with neural models.

**Parameters:**
- `model_id` (string): Model identifier
- `start_date` (string): Start date (YYYY-MM-DD)
- `end_date` (string): End date (YYYY-MM-DD)
- `benchmark` (string): Benchmark for comparison (default: "sp500")
- `rebalance_frequency` (string): Rebalancing frequency
- `use_gpu` (bool): Enable GPU acceleration

### neural_model_status
Get status and performance metrics for neural models.

**Parameters:**
- `model_id` (string, optional): Specific model ID (if not provided, returns all models)

### neural_optimize
Optimize neural model hyperparameters.

**Parameters:**
- `model_id` (string): Model identifier
- `parameter_ranges` (dict): Parameter ranges for optimization
- `trials` (int): Number of optimization trials (default: 100)
- `optimization_metric` (string): Metric to optimize (default: "mae")
- `use_gpu` (bool): Enable GPU acceleration

## Trading Strategy Tools (4 tools)

### mcp__ai-news-trader__quick_analysis
Get real-time market analysis with neural enhancement.

**Parameters:**
- `symbol` (string): Trading symbol
- `use_gpu` (bool): Enable GPU acceleration for faster analysis

**Native Claude Code Example:**
```
"Run a quick market analysis on AAPL using GPU acceleration"
```

**Direct Tool Call:**
```json
{
  "tool": "mcp__ai-news-trader__quick_analysis",
  "parameters": {
    "symbol": "AAPL",
    "use_gpu": true
  }
}
```

**Expected Return:**
```json
{
  "symbol": "AAPL",
  "current_price": 185.50,
  "technical_indicators": {
    "rsi": 58.3,
    "macd": "bullish",
    "moving_avg_20": 183.25
  },
  "sentiment": "positive",
  "recommendation": "hold",
  "gpu_time_ms": 12.5
}
```

### simulate_trade
Simulate trading operations with performance tracking.

**Parameters:**
- `strategy` (string): Trading strategy name
- `symbol` (string): Trading symbol
- `action` (string): Trade action ("buy" or "sell")
- `use_gpu` (bool): Enable GPU-optimized execution

### execute_trade
Execute live trades with advanced order management (demo mode).

**Parameters:**
- `strategy` (string): Trading strategy
- `symbol` (string): Trading symbol
- `action` (string): Trade action
- `quantity` (int): Number of shares/units
- `order_type` (string): Order type ("market" or "limit")
- `limit_price` (float, optional): Limit price for limit orders

### mcp__ai-news-trader__get_portfolio_status
Get comprehensive portfolio status with analytics.

**Parameters:**
- `include_analytics` (bool): Include advanced analytics (default: true)

**Native Claude Code Example:**
```
"Show my current portfolio status with full analytics"
```

**Direct Tool Call:**
```json
{
  "tool": "mcp__ai-news-trader__get_portfolio_status",
  "parameters": {
    "include_analytics": true
  }
}
```

**Expected Return:**
```json
{
  "total_value": 125000.00,
  "positions": [
    {"symbol": "AAPL", "shares": 100, "value": 18550.00},
    {"symbol": "TSLA", "shares": 50, "value": 12500.00}
  ],
  "analytics": {
    "sharpe_ratio": 2.15,
    "max_drawdown": -8.5,
    "total_return_pct": 15.3,
    "volatility": 0.18
  }
}
```

## Advanced Analytics Tools (7 tools)

### run_backtest
Run comprehensive historical backtesting.

**Parameters:**
- `strategy` (string): Trading strategy name
- `symbol` (string): Trading symbol
- `start_date` (string): Start date
- `end_date` (string): End date
- `use_gpu` (bool): Enable GPU acceleration
- `benchmark` (string): Benchmark comparison
- `include_costs` (bool): Include transaction costs

### optimize_strategy
Optimize trading strategy parameters.

**Parameters:**
- `strategy` (string): Strategy name
- `symbol` (string): Trading symbol
- `parameter_ranges` (dict): Parameter ranges for optimization
- `max_iterations` (int): Maximum iterations (default: 1000)
- `use_gpu` (bool): Enable GPU acceleration
- `optimization_metric` (string): Metric to optimize

### performance_report
Generate detailed performance analytics.

**Parameters:**
- `strategy` (string): Strategy name
- `period_days` (int): Analysis period in days (default: 30)
- `include_benchmark` (bool): Include benchmark comparison
- `use_gpu` (bool): Enable GPU acceleration

### correlation_analysis
Analyze asset correlations with GPU acceleration.

**Parameters:**
- `symbols` (list): List of trading symbols
- `period_days` (int): Analysis period (default: 90)
- `use_gpu` (bool): Enable GPU acceleration

### run_benchmark
Run comprehensive benchmarks for strategy performance.

**Parameters:**
- `strategy` (string): Strategy name
- `benchmark_type` (string): Type of benchmark ("performance" or "system")
- `use_gpu` (bool): Enable GPU acceleration

### risk_analysis
Comprehensive portfolio risk analysis.

**Parameters:**
- `portfolio` (list): Portfolio positions
- `var_confidence` (float): VaR confidence level (default: 0.05)
- `time_horizon` (int): Time horizon in days
- `use_monte_carlo` (bool): Use Monte Carlo simulation
- `use_gpu` (bool): Enable GPU acceleration

## News & Sentiment Tools (2 tools)

### mcp__ai-news-trader__analyze_news
AI-powered news sentiment analysis.

**Parameters:**
- `symbol` (string): Trading symbol
- `lookback_hours` (int): Hours to look back (default: 24)
- `sentiment_model` (string): Sentiment model to use (default: "enhanced")
- `use_gpu` (bool): Enable GPU acceleration for NLP

**Native Claude Code Example:**
```
"Analyze TSLA news sentiment for the past 48 hours using GPU"
```

**Direct Tool Call:**
```json
{
  "tool": "mcp__ai-news-trader__analyze_news",
  "parameters": {
    "symbol": "TSLA",
    "lookback_hours": 48,
    "sentiment_model": "enhanced",
    "use_gpu": true
  }
}
```

**Expected Return:**
```json
{
  "symbol": "TSLA",
  "overall_sentiment": 0.72,
  "sentiment_label": "positive",
  "news_count": 42,
  "key_themes": ["earnings beat", "new product launch", "market expansion"],
  "confidence": 0.89,
  "gpu_time_ms": 156.2
}
```

### get_news_sentiment
Get real-time news sentiment data.

**Parameters:**
- `symbol` (string): Trading symbol
- `sources` (list, optional): News sources to analyze

## System Tools (2 tools)

### ping
Simple connectivity test for the MCP server.

**Parameters:** None

### mcp__ai-news-trader__list_strategies
List all available trading strategies with GPU capabilities.

**Parameters:** None

**Native Claude Code Example:**
```
"Show me all available trading strategies"
```

**Direct Tool Call:**
```json
{
  "tool": "mcp__ai-news-trader__list_strategies",
  "parameters": {}
}
```

**Expected Return:**
```json
{
  "strategies": [
    {"name": "momentum", "gpu_enabled": true, "type": "trend_following"},
    {"name": "mean_reversion", "gpu_enabled": true, "type": "contrarian"},
    {"name": "neural_enhanced", "gpu_enabled": true, "type": "ai_driven"},
    {"name": "arbitrage", "gpu_enabled": false, "type": "market_neutral"}
  ]
}
```

## Native MCP Usage Patterns

### Neural-Enhanced Trading Workflow
```
# Just ask Claude naturally:
"I need to analyze AAPL for trading. Please:
1. Generate a 24-hour neural forecast
2. Analyze recent news sentiment
3. Run a quick market analysis
4. Simulate a momentum trade
5. Show me the performance metrics"
```

### Strategy Development and Testing
```
# Natural language request:
"Help me test the momentum strategy:
1. Show available strategies first
2. Backtest momentum on AAPL for 2024 Q1-Q2
3. Optimize the parameters for best Sharpe ratio
4. Run risk analysis on the results
5. Generate a performance report"
```

### Portfolio Management Workflow
```
# Simple request to Claude:
"Analyze my portfolio:
1. Show current status with analytics
2. Check correlations between holdings
3. Analyze news sentiment for all positions
4. Run Monte Carlo risk simulation
5. Generate neural forecasts for each holding"
```

### Direct Tool Chaining Example
```python
# Claude can chain multiple MCP tools automatically:
"First check AAPL sentiment, then if positive, generate a neural forecast and simulate a buy trade"

# Claude will execute:
# 1. mcp__ai-news-trader__analyze_news(symbol="AAPL")
# 2. If sentiment > 0.6:
#    - mcp__ai-news-trader__neural_forecast(symbol="AAPL", horizon=24)
#    - mcp__ai-news-trader__simulate_trade(symbol="AAPL", action="buy")
```

## Performance Notes

### GPU Acceleration
- Claude automatically enables GPU when beneficial
- You can explicitly request: "Use GPU acceleration for faster results"
- GPU provides 6,250x speedup for:
  - Neural forecasting operations
  - Market analysis and backtesting
  - News sentiment analysis (NLP)
  - Monte Carlo simulations
  - Correlation calculations

### Example GPU Request:
```
"Analyze AAPL news sentiment with GPU acceleration for fastest results"
```

### Tool Combinations
Combine multiple tools for comprehensive analysis:
- `neural_forecast` + `analyze_news` for sentiment-enhanced predictions
- `quick_analysis` + `run_backtest` for strategy validation
- `optimize_strategy` + `risk_analysis` for robust optimization
- `correlation_analysis` + `neural_forecast` for multi-asset forecasting

## Error Handling
All tools include comprehensive error handling:
- Validation of input parameters
- Graceful fallbacks for missing data
- GPU availability checking
- Model existence validation
- Data format verification

## Best Practices for Native MCP Usage

### Natural Language Requests
1. **Be specific but conversational:**
   - ✅ "Analyze TSLA sentiment for the past 2 days"
   - ❌ "mcp__ai-news-trader__analyze_news symbol TSLA lookback_hours 48"

2. **Chain operations naturally:**
   - "Check AAPL sentiment, and if positive, generate a forecast"
   - "Analyze my portfolio and suggest rebalancing based on correlations"

3. **Request GPU when needed:**
   - "Run neural forecast for AAPL with GPU for fastest results"
   - "Use GPU to analyze correlations across 50 stocks"

### Common Patterns
```
# Morning routine
"Good morning! Please:
1. Check my portfolio status
2. Analyze overnight news for my holdings
3. Generate neural forecasts for any stocks with positive sentiment
4. Suggest any trading opportunities"

# Research request
"I'm interested in NVDA. Can you:
1. Run a comprehensive analysis
2. Check recent sentiment
3. Generate a 48-hour forecast
4. Compare with sector performance"

# Risk check
"Perform a full risk analysis of my portfolio using Monte Carlo simulation with GPU"
```

### Error Handling
Claude handles errors gracefully:
- Missing data: Provides alternative analysis
- Tool failures: Suggests workarounds
- Invalid parameters: Asks for clarification

### Tips for Best Results
1. **Start simple:** "Analyze AAPL" before complex workflows
2. **Build incrementally:** Add requirements as you see results
3. **Let Claude optimize:** It will choose the best tools and parameters
4. **Ask for explanations:** "Explain the forecast confidence interval"
5. **Request specific formats:** "Show results as a table" or "Give me JSON output"

## Further Reading
For detailed documentation on individual MCP tools, including advanced usage patterns, integration examples, and best practices, see the [`mcp-tools/`](./mcp-tools/) directory:

- **[Complete Tool Index](./mcp-tools/index.md)** - All 21 tools with quick reference
- **[News Analysis Tools](./mcp-tools/analyze-news.md)** - Deep dive into AI sentiment analysis
- **[Real-time Sentiment](./mcp-tools/get-news-sentiment.md)** - Fast sentiment monitoring
- More tool-specific guides coming soon!