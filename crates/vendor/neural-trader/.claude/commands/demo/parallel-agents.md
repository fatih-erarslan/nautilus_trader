# Claude Code Demo: Parallel Trading Agents

This demo runs 5 specialized AI trading agents in parallel to showcase the platform's comprehensive capabilities.

## Instructions

Copy and paste these prompts into Claude Code to launch parallel agents:

### Agent 1: Market Analysis
```
I need you to act as a Market Analysis Agent. Please:
1. Use mcp__ai-news-trader__quick_analysis to analyze AAPL with use_gpu: true
2. Use mcp__ai-news-trader__quick_analysis to analyze NVDA with use_gpu: true  
3. Use mcp__ai-news-trader__neural_forecast for NVDA with horizon: 7, confidence_level: 0.95, use_gpu: true
4. Use mcp__ai-news-trader__get_system_metrics with metrics: ["cpu", "memory", "latency", "throughput"], include_history: true

Summarize the market conditions, forecast, and system performance.
```

### Agent 2: News Sentiment Analysis
```
I need you to act as a News Sentiment Analyst. Please:
1. Use mcp__ai-news-trader__control_news_collection with action: "start", symbols: ["AAPL", "NVDA", "TSLA", "GOOGL", "MSFT"], update_frequency: 300, lookback_hours: 48
2. Use mcp__ai-news-trader__analyze_news for TSLA with lookback_hours: 48, sentiment_model: "enhanced", use_gpu: true
3. Use mcp__ai-news-trader__get_news_trends with symbols: ["AAPL", "TSLA", "NVDA"], time_intervals: [1, 6, 24, 48]
4. Use mcp__ai-news-trader__fetch_filtered_news with symbols: ["AAPL", "NVDA", "TSLA", "GOOGL", "MSFT"], sentiment_filter: "positive", relevance_threshold: 0.8, limit: 10

Report on overall market sentiment and key opportunities.
```

### Agent 3: Strategy Optimization
```
I need you to act as a Strategy Optimization Agent. Please:
1. Use mcp__ai-news-trader__list_strategies to see all available strategies
2. Use mcp__ai-news-trader__get_strategy_comparison with strategies: ["momentum_trading_optimized", "swing_trading_optimized", "mean_reversion_optimized"], metrics: ["sharpe_ratio", "total_return", "max_drawdown", "win_rate"]
3. Use mcp__ai-news-trader__run_backtest with strategy: "momentum_trading_optimized", symbol: "NVDA", start_date: "2024-05-28", end_date: "2025-06-28", use_gpu: true
4. Use mcp__ai-news-trader__adaptive_strategy_selection with symbol: "AAPL", auto_switch: false

Recommend the best strategy based on current conditions.
```

### Agent 4: Risk Management
```
I need you to act as a Risk Management Agent. Please:
1. Use mcp__ai-news-trader__get_portfolio_status with include_analytics: true
2. Use mcp__ai-news-trader__cross_asset_correlation_matrix with assets: ["AAPL", "NVDA", "TSLA", "GOOGL", "MSFT"], lookback_days: 90, include_prediction_confidence: true
3. Use mcp__ai-news-trader__risk_analysis with portfolio: [{"symbol": "AAPL", "shares": 100, "entry_price": 185.0}, {"symbol": "NVDA", "shares": 50, "entry_price": 450.0}, {"symbol": "GOOGL", "shares": 75, "entry_price": 140.0}], time_horizon: 5, use_monte_carlo: true, use_gpu: true
4. Use mcp__ai-news-trader__monitor_strategy_health with strategy: "momentum_trading_optimized"

Provide risk assessment and recommendations.
```

### Agent 5: Trading Execution
```
I need you to act as a Trading Execution Agent. Please:
1. Use mcp__ai-news-trader__simulate_trade with strategy: "momentum_trading_optimized", symbol: "NVDA", action: "buy", use_gpu: true
2. Use mcp__ai-news-trader__get_prediction_markets_tool with category: "Crypto", sort_by: "volume", limit: 5
3. Use mcp__ai-news-trader__calculate_expected_value_tool with market_id: "crypto_btc_100k", investment_amount: 1000, confidence_adjustment: 1.1, use_gpu: true
4. Use mcp__ai-news-trader__performance_report with strategy: "momentum_trading_optimized", period_days: 30, include_benchmark: true, use_gpu: true

Summarize trading opportunities and performance.
```

## Running All Agents in Parallel

To run all 5 agents simultaneously in Claude Code:

1. Open a new conversation
2. Copy all 5 agent prompts above
3. Paste them all at once (Claude will process them in parallel)
4. Wait for comprehensive results from all agents

## Expected Results

- **Market Analysis**: Real-time prices, technical indicators, 7-day forecasts
- **News Sentiment**: Aggregated sentiment scores, trending news, opportunities
- **Strategy Optimization**: Performance comparisons, backtest results, recommendations
- **Risk Management**: Portfolio VaR, correlations, risk warnings
- **Trading Execution**: Trade simulations, prediction markets, performance metrics

## Tips

- Enable GPU (`use_gpu: true`) for 1000x faster processing
- Run during market hours for real-time data
- Combine insights from all agents for best decisions
- Monitor system metrics to ensure optimal performance