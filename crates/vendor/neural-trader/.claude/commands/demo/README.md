# Claude Code Trading Platform - Demo Commands

Welcome to the AI News Trading Platform demo commands for Claude Code. These demos show you how to use all 41 MCP tools directly in Claude Code.

## 🚀 Quick Start

Simply copy any prompt from the demo files and paste it into Claude Code. The platform will execute the MCP tools and provide results.

## 📁 Available Demos

### 1. [Parallel Agents](parallel-agents.md)
Run 5 specialized trading agents simultaneously to see the full platform capabilities.
- Market Analysis Agent
- News Sentiment Agent  
- Strategy Optimization Agent
- Risk Management Agent
- Trading Execution Agent

### 2. [Market Analysis](market-analysis.md)
Learn real-time market analysis with AI neural forecasting.
- Quick stock analysis with technical indicators
- 7-30 day AI price predictions
- System performance monitoring
- Multi-symbol comparison

### 3. [News Sentiment](news-sentiment.md)
Master news aggregation and sentiment analysis.
- Multi-source news collection
- AI sentiment scoring with FinBERT
- Trend analysis across timeframes
- High-impact news filtering

### 4. [Strategy Optimization](strategy-optimization.md)
Compare, backtest, and optimize trading strategies.
- Strategy performance comparison
- Historical backtesting with GPU
- Parameter optimization
- Adaptive strategy selection

### 5. [Risk Management](risk-management.md)
Advanced portfolio risk analysis and management.
- Portfolio VaR calculations
- Correlation analysis
- Monte Carlo simulations
- Optimal rebalancing

### 6. [Trading Execution](trading-execution.md)
Execute trades and analyze prediction markets.
- Trade simulation
- Multi-asset batch execution
- Prediction market analysis
- Performance tracking

## 🎯 How to Use

### Method 1: Copy Individual Prompts
1. Open any demo file (e.g., `market-analysis.md`)
2. Find a prompt block in gray background
3. Copy the entire prompt
4. Paste into Claude Code
5. Claude will execute the MCP tools and show results

### Method 2: Run Complete Workflows
1. Open `parallel-agents.md`
2. Copy all 5 agent prompts
3. Paste them together in Claude Code
4. Get comprehensive analysis from all agents

### Method 3: Build Custom Workflows
1. Combine prompts from different demos
2. Modify parameters to fit your needs
3. Create your own trading strategies

## 📊 MCP Tool Reference

All tools use the prefix: `mcp__ai-news-trader__`

### Most Used Tools:
- `quick_analysis` - Real-time market analysis
- `neural_forecast` - AI price predictions
- `analyze_news` - Sentiment analysis
- `run_backtest` - Strategy testing
- `risk_analysis` - Portfolio risk metrics
- `simulate_trade` - Trade simulation

## 💡 Pro Tips

### Enable GPU Acceleration
Always add `use_gpu: true` to parameters for 1000x faster processing:
```
Use mcp__ai-news-trader__neural_forecast with:
- symbol: "AAPL"
- horizon: 7
- use_gpu: true  # ← Always include this
```

### Batch Operations
Process multiple symbols efficiently:
```
For each symbol in [AAPL, NVDA, TSLA, GOOGL, MSFT]:
  Use mcp__ai-news-trader__quick_analysis with:
  - symbol: [symbol]
  - use_gpu: true
```

### Combine Tools
Best results come from combining multiple tools:
```
1. First check sentiment with analyze_news
2. Then get technical analysis with quick_analysis  
3. Finally generate forecast with neural_forecast
4. Make decision based on all three signals
```

## 🔧 Troubleshooting

### "Tool not found"
- Ensure you're using exact tool name with prefix
- Check spelling: `mcp__ai-news-trader__tool_name`

### Slow Performance
- Add `use_gpu: true` to parameters
- Reduce number of symbols in batch operations
- Check system metrics for resource usage

### No Data
- Markets may be closed (mock data will be used)
- Symbol might be invalid (use standard tickers)
- Time range might be too far in past

## 📈 Example Complete Workflow

Here's a full trading decision workflow:

```
I need to make a trading decision for AAPL. Please:

1. Use mcp__ai-news-trader__quick_analysis with symbol: "AAPL", use_gpu: true
2. Use mcp__ai-news-trader__analyze_news with symbol: "AAPL", lookback_hours: 48, sentiment_model: "enhanced", use_gpu: true
3. Use mcp__ai-news-trader__neural_forecast with symbol: "AAPL", horizon: 7, confidence_level: 0.95, use_gpu: true
4. Use mcp__ai-news-trader__risk_analysis with portfolio including AAPL position

Based on all analyses, provide a clear BUY/HOLD/SELL recommendation with:
- Entry price
- Stop loss
- Take profit
- Position size
- Risk/reward ratio
```

## 🚦 Getting Started

1. Start with [`parallel-agents.md`](parallel-agents.md) to see all capabilities
2. Try [`market-analysis.md`](market-analysis.md) for basic usage
3. Explore other demos based on your interests
4. Combine tools to create your own strategies

Ready to begin? Open any demo file and start trading with AI!