# Quick Reference - Native MCP Trading Tools

## Essential MCP Tools for Trading

### 🚀 Quick Start with Claude Code
```
"Let's start trading:
1. Generate an AI price prediction for AAPL
2. Check my portfolio status
3. Analyze market sentiment
4. Show me trading opportunities"
```

### 🤖 Direct MCP Tool Access
All tools use the prefix: `mcp__ai-news-trader__<tool_name>`

**Example Natural Requests:**
- "Use neural_forecast to predict AAPL prices"
- "Run quick_analysis on TSLA"
- "Check portfolio status with analytics"
- "Analyze news sentiment for tech stocks"

### 🧠 Neural Forecasting Tools (6 MCP tools)

**mcp__ai-news-trader__neural_forecast**
```
"Generate 24-hour forecast for AAPL with GPU"
"Predict TSLA prices for next 48 hours"
"Forecast my entire portfolio"
```

**mcp__ai-news-trader__neural_train**
```
"Train a neural model on trading_data.csv"
"Create Transformer model with 200 epochs"
```

**mcp__ai-news-trader__neural_evaluate**
```
"Evaluate my model on test data"
"Check model accuracy metrics"
```

**mcp__ai-news-trader__neural_backtest**
```
"Backtest model on 2024 data"
"Historical validation for AAPL"
```

**mcp__ai-news-trader__neural_model_status**
```
"Show all neural model status"
"Check if models are healthy"
```

**mcp__ai-news-trader__neural_optimize**
```
"Optimize model hyperparameters"
"Improve forecast accuracy"
```

### 📈 Trading Strategy Tools (4 MCP tools)

**mcp__ai-news-trader__quick_analysis**
```
"Analyze AAPL market conditions"
"Quick check on my watchlist stocks"
"Is TSLA good to buy now?"
```

**mcp__ai-news-trader__simulate_trade**
```
"Simulate buying 100 AAPL shares"
"Test momentum strategy on TSLA"
"Simulate portfolio rebalancing"
```

**mcp__ai-news-trader__execute_trade**
```
"Buy 50 shares of NVDA at market"
"Place limit order for AAPL at $185"
"Execute the simulated trades"
```

**mcp__ai-news-trader__get_portfolio_status**
```
"Show my portfolio with analytics"
"Current positions and performance"
"Portfolio risk metrics"
```

### ⚡ Advanced Analytics Tools (7 MCP tools)

**mcp__ai-news-trader__run_backtest**
```
"Backtest momentum strategy on AAPL for 2024"
"Test strategy with transaction costs"
"Compare against S&P 500 benchmark"
```

**mcp__ai-news-trader__optimize_strategy**
```
"Optimize momentum parameters for best Sharpe"
"Find best settings for mean reversion"
"Maximize returns with limited drawdown"
```

**mcp__ai-news-trader__performance_report**
```
"Generate 30-day performance report"
"Show detailed strategy metrics"
"Performance breakdown by asset"
```

**mcp__ai-news-trader__risk_analysis**
```
"Analyze portfolio risk with Monte Carlo"
"Calculate VaR at 95% confidence"
"Show all risk metrics"
```

**mcp__ai-news-trader__correlation_analysis**
```
"Analyze FAANG stock correlations"
"Find uncorrelated assets"
"Portfolio correlation matrix"
```

**mcp__ai-news-trader__run_benchmark**
```
"Benchmark all strategies"
"Compare strategy speeds"
"Performance vs market"
```

### 📰 News & Sentiment Tools (2 MCP tools)

**mcp__ai-news-trader__analyze_news**
```
"Analyze TSLA news sentiment for 48 hours"
"Check market sentiment for tech sector"
"AI analysis of recent Fed announcements"
```

**mcp__ai-news-trader__get_news_sentiment**
```
"Real-time sentiment for AAPL"
"Current market mood"
"Breaking news impact on my stocks"
```

### 🔧 System Tools (2 MCP tools)

**mcp__ai-news-trader__ping**
```
"Check if trading tools are working"
"Test MCP connection"
```

**mcp__ai-news-trader__list_strategies**
```
"Show all available trading strategies"
"Which strategies support GPU?"
"List AI-enhanced strategies"
```

## Complete 21 MCP Tools Reference

### Tool Categories & Usage

**Neural Forecasting (6 tools)**
- `mcp__ai-news-trader__neural_forecast` - AI price predictions
- `mcp__ai-news-trader__neural_train` - Train custom models
- `mcp__ai-news-trader__neural_evaluate` - Test model accuracy
- `mcp__ai-news-trader__neural_backtest` - Historical validation
- `mcp__ai-news-trader__neural_model_status` - Model health checks
- `mcp__ai-news-trader__neural_optimize` - Hyperparameter tuning

**Trading Strategy (4 tools)**
- `mcp__ai-news-trader__quick_analysis` - Market analysis
- `mcp__ai-news-trader__simulate_trade` - Trade simulation
- `mcp__ai-news-trader__execute_trade` - Order execution
- `mcp__ai-news-trader__get_portfolio_status` - Portfolio analytics

**Advanced Analytics (7 tools)**
- `mcp__ai-news-trader__run_backtest` - Strategy backtesting
- `mcp__ai-news-trader__optimize_strategy` - Parameter optimization
- `mcp__ai-news-trader__performance_report` - Performance metrics
- `mcp__ai-news-trader__risk_analysis` - Risk assessment
- `mcp__ai-news-trader__correlation_analysis` - Asset correlations
- `mcp__ai-news-trader__run_benchmark` - Strategy benchmarking

**News & Sentiment (2 tools)**
- `mcp__ai-news-trader__analyze_news` - AI sentiment analysis
- `mcp__ai-news-trader__get_news_sentiment` - Real-time sentiment

**System Tools (2 tools)**
- `mcp__ai-news-trader__ping` - Connection test
- `mcp__ai-news-trader__list_strategies` - Available strategies

## Common MCP Workflows

### 📊 Daily Trading Workflow
```
"Good morning! Please help with my daily trading routine:
1. Check my portfolio status (get_portfolio_status)
2. Generate AI forecasts for AAPL, MSFT, GOOGL (neural_forecast)
3. Analyze overnight news sentiment (analyze_news)
4. Run market analysis on any movers (quick_analysis)
5. Simulate any recommended trades (simulate_trade)"
```

**Claude automatically chains the tools:**
1. `mcp__ai-news-trader__get_portfolio_status`
2. `mcp__ai-news-trader__neural_forecast` (multiple symbols)
3. `mcp__ai-news-trader__analyze_news` (for each holding)
4. `mcp__ai-news-trader__quick_analysis` (for opportunities)
5. `mcp__ai-news-trader__simulate_trade` (if needed)

### 🔬 Strategy Development Workflow
```
"Help me develop a momentum trading strategy:
1. Analyze historical momentum patterns (run_backtest)
2. Optimize parameters for best returns (optimize_strategy)
3. Test across different market conditions (run_backtest)
4. Compare with buy-and-hold (performance_report)
5. Assess risk metrics (risk_analysis)"
```

**Tool execution sequence:**
1. `mcp__ai-news-trader__run_backtest` (initial testing)
2. `mcp__ai-news-trader__optimize_strategy` (find best params)
3. `mcp__ai-news-trader__run_backtest` (validate optimized)
4. `mcp__ai-news-trader__performance_report` (detailed metrics)
5. `mcp__ai-news-trader__risk_analysis` (risk assessment)

### 🎯 Model Training Workflow
```
"Train a new neural forecasting model:
1. Train on my trading_data.csv with GPU (neural_train)
2. Evaluate accuracy on test set (neural_evaluate)
3. Backtest on historical data (neural_backtest)
4. Check model status (neural_model_status)
5. Optimize if needed (neural_optimize)"
```

**Automated tool chain:**
1. `mcp__ai-news-trader__neural_train`
2. `mcp__ai-news-trader__neural_evaluate`
3. `mcp__ai-news-trader__neural_backtest`
4. `mcp__ai-news-trader__neural_model_status`
5. `mcp__ai-news-trader__neural_optimize` (if accuracy < threshold)

### 📈 Portfolio Management Workflow
```
"Analyze and optimize my portfolio:
1. Show current status with metrics (get_portfolio_status)
2. Check asset correlations (correlation_analysis)
3. Run risk analysis with Monte Carlo (risk_analysis)
4. Suggest rebalancing for lower risk
5. Simulate the rebalancing trades (simulate_trade)"
```

**Claude executes:**
1. `mcp__ai-news-trader__get_portfolio_status`
2. `mcp__ai-news-trader__correlation_analysis`
3. `mcp__ai-news-trader__risk_analysis`
4. Analysis and recommendations
5. `mcp__ai-news-trader__simulate_trade` (for each change)

## Quick MCP Patterns

### Morning Check
```
"Morning trading check:
1. Portfolio status
2. Overnight news
3. Pre-market movers
4. AI forecasts for holdings"
```

### Opportunity Hunt
```
"Find trading opportunities:
1. Scan for oversold stocks
2. Check positive news sentiment
3. Generate forecasts for candidates
4. Simulate potential trades"
```

### Risk Check
```
"Quick risk assessment:
1. Current portfolio risk metrics
2. Correlation check
3. VaR calculation
4. Suggest adjustments"
```

### End of Day
```
"End of day summary:
1. Today's performance
2. Executed trades review
3. Tomorrow's forecast
4. Any risks to monitor"
```

## MCP Best Practices

### Natural Language Tips
1. **Be conversational:** "Help me analyze AAPL"
2. **Chain requests:** "If sentiment is positive, then forecast"
3. **Specify preferences:** "Use GPU for faster results"
4. **Ask for explanations:** "Explain the risk metrics"

### Performance Optimization
- Claude automatically uses GPU when beneficial
- Tools are chained efficiently
- Results are cached when appropriate
- Parallel execution for multiple symbols

### Common Patterns
```
# Research first
"Before trading NVDA, analyze sentiment and forecast"

# Risk management
"Only suggest trades that won't increase portfolio risk"

# Conditional logic
"If TSLA forecast is bullish, simulate a buy trade"

# Comprehensive analysis
"Full analysis: sentiment + technical + forecast + risk"
```

## Getting Help with MCP Tools

### Ask Claude Directly
```
"How do I use the neural forecasting tools?"
"What's the difference between quick_analysis and run_backtest?"
"Show me an example of portfolio risk analysis"
"Help me optimize my trading strategy"
```

### Tool Discovery
```
"What MCP tools are available for trading?"
"Which tools help with risk management?"
"Show me all neural forecasting tools"
"What tools work with GPU acceleration?"
```

### Troubleshooting
```
"The forecast tool isn't working - help!"
"How do I check if MCP tools are connected?"
"Why is my backtest taking so long?"
"Debug my portfolio analysis request"
```

### Learning Patterns
```
"Show me a complete trading workflow"
"Best practices for using MCP tools"
"Common mistakes to avoid"
"Advanced MCP tool combinations"
```