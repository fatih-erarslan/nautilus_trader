# AI News Trading Platform - Parallel Agent Execution

## 🚀 Launching 5 Parallel Trading Agents

This demo showcases the platform's capabilities by running 5 specialized agents in parallel.
Each agent uses different MCP tools to analyze markets, news, strategies, risk, and execute trades.

### Configuration:
- **Symbols**: AAPL, NVDA, TSLA, GOOGL, MSFT
- **Strategies**: momentum_trading_optimized, swing_trading_optimized  
- **Timeframe**: Last 30 days
- **GPU**: Enabled for all operations

### Agents Being Launched:

1. **Market Analyst** - Real-time market analysis and AI forecasting
2. **News Analyst** - Multi-source news aggregation and sentiment analysis
3. **Strategy Optimizer** - Strategy comparison and parameter optimization
4. **Risk Manager** - Portfolio risk analysis and correlation studies
5. **Trader** - Trade execution and prediction market analysis

---

## Execution Instructions

To run this demo in Claude Code, use the Task tool to launch agents in parallel:

```
# Launch all 5 agents simultaneously
Use the Task tool 5 times in a single message with these prompts:
```

### Agent 1: Market Analyst
```
Description: Market Analyst - Analyze markets using MCP tools
Prompt: You are a Market Analysis Agent. Your task is to analyze market conditions using MCP tools.

TODO List:
1. Analyze current market conditions for symbols: AAPL, NVDA, TSLA
2. Generate 7-day neural forecasts for top performers
3. Check system performance metrics
4. Create market assessment summary

Use these MCP tools in sequence:
1. mcp__ai-news-trader__quick_analysis for each symbol (AAPL, NVDA, TSLA) with use_gpu: true
2. mcp__ai-news-trader__neural_forecast for NVDA with horizon: 7, confidence_level: 0.95, use_gpu: true
3. mcp__ai-news-trader__get_system_metrics with metrics: ["cpu", "memory", "latency", "throughput"]

Provide a concise summary of:
- Current price trends and technical indicators
- AI forecast predictions with confidence intervals
- System performance status
```

### Agent 2: News Analyst
```
Description: News Analyst - Analyze markets using MCP tools
Prompt: You are a News Sentiment Analyst. Your task is to analyze news and sentiment using MCP tools.

TODO List:
1. Start news collection for symbols: AAPL, NVDA, TSLA, GOOGL, MSFT
2. Analyze sentiment trends over multiple timeframes
3. Filter high-relevance positive news
4. Report sentiment momentum changes

Use these MCP tools:
1. mcp__ai-news-trader__control_news_collection with action: "start", symbols: ['AAPL', 'NVDA', 'TSLA', 'GOOGL', 'MSFT'], lookback_hours: 48
2. mcp__ai-news-trader__analyze_news for TSLA with lookback_hours: 48, sentiment_model: "enhanced", use_gpu: true
3. mcp__ai-news-trader__get_news_trends with symbols: ["AAPL", "TSLA", "NVDA"], time_intervals: [1, 6, 24, 48]
4. mcp__ai-news-trader__fetch_filtered_news with sentiment_filter: "positive", relevance_threshold: 0.8

Summarize:
- Overall market sentiment
- Key news drivers
- Sentiment momentum trends
```

### Agent 3: Strategy Optimizer
```
Description: Strategy Optimizer - Analyze markets using MCP tools
Prompt: You are a Strategy Optimization Agent. Your task is to optimize trading strategies using MCP tools.

TODO List:
1. Compare available trading strategies
2. Run backtest on best performing symbol
3. Optimize strategy parameters
4. Recommend adaptive strategy

Use these MCP tools:
1. mcp__ai-news-trader__list_strategies to see all available strategies
2. mcp__ai-news-trader__get_strategy_comparison with strategies: ['momentum_trading_optimized', 'swing_trading_optimized'], metrics: ["sharpe_ratio", "total_return", "max_drawdown"]
3. mcp__ai-news-trader__run_backtest with strategy: "momentum_trading_optimized", symbol: "NVDA", start_date: "2025-05-29", end_date: "2025-06-28", use_gpu: true
4. mcp__ai-news-trader__adaptive_strategy_selection with symbol: "AAPL"

Report:
- Best performing strategy
- Backtest results summary
- Optimization recommendations
```

### Agent 4: Risk Manager
```
Description: Risk Manager - Analyze markets using MCP tools
Prompt: You are a Risk Management Agent. Your task is to analyze portfolio risk using MCP tools.

TODO List:
1. Get current portfolio status
2. Calculate asset correlations
3. Run Monte Carlo risk simulation
4. Monitor strategy health

Use these MCP tools:
1. mcp__ai-news-trader__get_portfolio_status with include_analytics: true
2. mcp__ai-news-trader__cross_asset_correlation_matrix with assets: ['AAPL', 'NVDA', 'TSLA', 'GOOGL', 'MSFT'], lookback_days: 90, include_prediction_confidence: true
3. mcp__ai-news-trader__risk_analysis with portfolio sample (AAPL: 100 shares, NVDA: 50 shares, GOOGL: 75 shares), time_horizon: 5, use_monte_carlo: true, use_gpu: true
4. mcp__ai-news-trader__monitor_strategy_health with strategy: "momentum_trading_optimized"

Provide:
- Portfolio risk metrics (VaR, CVaR)
- Correlation insights
- Risk alerts or warnings
```

### Agent 5: Trader
```
Description: Trader - Analyze markets using MCP tools
Prompt: You are a Trading Execution Agent. Your task is to execute trades and analyze markets using MCP tools.

TODO List:
1. Simulate high-conviction trades
2. Analyze prediction markets
3. Calculate expected values
4. Generate performance report

Use these MCP tools:
1. mcp__ai-news-trader__simulate_trade with strategy: "momentum_trading_optimized", symbol: "NVDA", action: "buy", use_gpu: true
2. mcp__ai-news-trader__get_prediction_markets_tool with category: "Crypto", sort_by: "volume", limit: 5
3. mcp__ai-news-trader__calculate_expected_value_tool with market_id: "crypto_btc_100k", investment_amount: 1000, use_gpu: true
4. mcp__ai-news-trader__performance_report with strategy: "momentum_trading_optimized", period_days: 30, include_benchmark: true

Report:
- Trade simulation results
- Top prediction market opportunities
- Performance summary
```

## Expected Results

After running all agents in parallel, you'll receive:

### From Market Analyst:
- Current prices and technical indicators for AAPL, NVDA, TSLA
- 7-day AI forecast for NVDA with confidence intervals
- System performance metrics and GPU utilization

### From News Analyst:
- Real-time news collection status for all 5 symbols
- TSLA sentiment analysis with enhanced AI model
- Sentiment trends over [1, 6, 24, 48] hour periods
- High-relevance positive news filtered by 0.8 threshold

### From Strategy Optimizer:
- Comparison of momentum vs swing trading strategies
- 30-day backtest results for NVDA
- Adaptive strategy recommendation for AAPL
- Optimization insights and parameter suggestions

### From Risk Manager:
- Current portfolio value and positions
- 5x5 correlation matrix with ML confidence scores
- Monte Carlo VaR/CVaR (10,000 simulations)
- Strategy health score and alerts

### From Trader:
- NVDA buy trade simulation results
- Top 5 crypto prediction markets by volume
- Expected value calculation for BTC 100k market
- 30-day performance report with attribution

## Performance Benchmarks

With GPU acceleration enabled:
- Neural forecasts: ~0.1 seconds (1000x faster)
- Risk analysis: ~0.2 seconds (500x faster)
- Backtesting: ~1 second (100x faster)
- Sentiment analysis: ~0.5 seconds (50x faster)

Total execution time for all 5 agents: <5 seconds

---

Ready to run? Copy the agent prompts above and use them with Claude Code's Task tool!
