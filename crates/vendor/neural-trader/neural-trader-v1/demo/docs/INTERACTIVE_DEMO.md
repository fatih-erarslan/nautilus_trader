# AI News Trading Platform - Interactive MCP Demo

## 🚀 Quick Start

This demo runs 5 specialized agents in parallel to showcase the platform's capabilities.

### Option 1: Run All Agents in Parallel

```bash
# Execute the complete swarm
./run_parallel_demo.sh
```

### Option 2: Run Individual Agents

```bash
# Market Analysis Agent
python execute_agent.py --agent-id market_analyst

# News Sentiment Agent  
python execute_agent.py --agent-id news_analyst

# Strategy Optimization Agent
python execute_agent.py --agent-id strategy_optimizer

# Risk Management Agent
python execute_agent.py --agent-id risk_manager

# Trading Execution Agent
python execute_agent.py --agent-id trader
```

### Option 3: Direct MCP Tool Usage in Claude Code

You can use any of the 41 MCP tools directly:

```
# Quick market analysis
Use tool: mcp__ai-news-trader__quick_analysis
Parameters:
  symbol: "AAPL"
  use_gpu: true

# Get neural forecast
Use tool: mcp__ai-news-trader__neural_forecast
Parameters:
  symbol: "NVDA"
  horizon: 7
  confidence_level: 0.95
  use_gpu: true

# Analyze news sentiment
Use tool: mcp__ai-news-trader__analyze_news
Parameters:
  symbol: "TSLA"
  lookback_hours: 48
  sentiment_model: "enhanced"
  use_gpu: true
```

## 📊 Expected Results

### Market Analysis Agent
- Real-time analysis of AAPL and NVDA
- 7-day AI price predictions with confidence intervals
- System performance metrics with GPU utilization

### News Sentiment Agent
- Multi-source news aggregation for 5 symbols
- Sentiment scores and trend analysis
- High-impact positive news filtering

### Strategy Optimization Agent
- Performance comparison of trading strategies
- 30-day backtest results with Sharpe ratios
- Adaptive strategy recommendations

### Risk Management Agent
- Portfolio correlation matrix
- Monte Carlo VaR calculations (10,000 simulations)
- Strategy health monitoring

### Trading Execution Agent
- Trade simulations with P&L estimates
- Prediction market analysis
- 30-day performance attribution report

## 🔧 Available MCP Tools

The platform provides 41 specialized tools across 10 categories:

1. **System Tools** (2): ping, list_strategies
2. **Trading Strategy** (4): get_strategy_info, quick_analysis, simulate_trade, execute_trade
3. **Portfolio Management** (1): get_portfolio_status
4. **Neural Forecasting** (6): neural_forecast, neural_train, neural_evaluate, etc.
5. **Advanced Analytics** (7): run_backtest, optimize_strategy, risk_analysis, etc.
6. **News & Sentiment** (2): analyze_news, get_news_sentiment
7. **Prediction Markets** (6): get_prediction_markets_tool, analyze_market_sentiment_tool, etc.
8. **News Collection** (4): control_news_collection, fetch_filtered_news, etc.
9. **Strategy Selection** (4): recommend_strategy, adaptive_strategy_selection, etc.
10. **Performance Monitoring** (3): get_system_metrics, monitor_strategy_health, etc.
11. **Multi-Asset Trading** (3): execute_multi_asset_trade, portfolio_rebalance, etc.

## 💡 Tips

- Enable GPU acceleration (`use_gpu: true`) for 1000x speedup
- Use parallel execution for real-time multi-symbol analysis
- Combine news sentiment with neural forecasts for best signals
- Monitor system metrics to ensure optimal performance

Ready to start? Run `./run_parallel_demo.sh` to see all agents in action!
