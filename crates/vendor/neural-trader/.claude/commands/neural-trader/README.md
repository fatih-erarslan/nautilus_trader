# Claude Code Neural Trader - MCP Documentation Hub

## 🚀 Quick Start
The AI News Trading Platform provides 21 powerful MCP tools through Claude Code. This documentation shows you how to use them effectively for neural forecasting, trading, and portfolio management.

### Basic Usage
Simply tell Claude Code what you want:
```
"Generate a 24-hour forecast for AAPL"
"Analyze Tesla's news sentiment"
"Backtest momentum strategy on SPY"
```

## 📚 Documentation Index

### 1. [MCP Direct Usage](./mcp-direct-usage.md)
Learn how to call MCP tools directly in Claude Code
- Direct syntax examples for all 21 tools
- Natural language alternatives
- Quick reference card

### 2. [Claude Code Quick Start](./mcp-claude-code-quickstart.md)
Get started with MCP integration in minutes
- Server configuration details
- Basic examples
- Common workflows

### 3. [Multi-Agent Demonstrations](./swarm-demo.md)
See how Claude Code orchestrates complex multi-tool workflows
- Parallel analysis patterns
- Strategy development workflows
- Risk management automation

### 4. [MCP Tools Reference](./mcp-tools-reference.md)
Complete reference for all 21 trading tools
- Detailed parameters
- Expected outputs
- Best practices

### 5. [Neural Trading Workflows](./neural-trading-workflows.md)
Production-ready trading workflows
- Complete trading pipelines
- Real-time decision flows
- Portfolio optimization

### 6. [Trading Patterns](./mcp-trading-patterns.md)
Advanced patterns for sophisticated strategies
- Sequential patterns
- Parallel execution
- Conditional logic

### 7. [Trading Scenarios](./mcp-trading-scenarios.md)
Real-world trading scenarios
- Day trading
- Swing trading
- Portfolio management
- Crisis handling

### 8. [Automation Examples](./mcp-automation-examples.md)
Automate your trading workflows
- Daily routines
- Alert systems
- Strategy optimization
- Report generation

### 9. [Tool Combinations](./mcp-tool-combinations.md)
Powerful multi-tool strategies
- Analysis stacks
- Risk-aware trading
- Research pipelines

### 10. [Integration Guide](./mcp-integration-guide.md)
Advanced integration topics
- Memory persistence
- Error handling
- Production deployment
- Security practices

## 🚀 Real-Time Trading System

### Quick Start Commands
```bash
# Start real-time trading system
./start-real-time-trading start

# Run safe demo (no real trades)
./demo-real-time-trading 5

# Check system status
./start-real-time-trading status

# Stop system
./start-real-time-trading stop
```

### Background Process Management
The system runs as a persistent background process that:
- ✅ Monitors live WebSocket market data (40+ messages/second)
- ✅ Analyzes news sentiment in real-time with confidence scoring
- ✅ Generates trading signals automatically using combined analysis
- ✅ Executes trades with comprehensive risk management
- ✅ Provides continuous performance monitoring via Claude Code integration
- ✅ Automatically integrates with Claude Code's native background process system
- ✅ Returns bash_id for real-time monitoring with BashOutput tool

**Agent Support**: Use the `real-time-trading-agent` for automated management of trading operations and background process orchestration.

## 🎯 Common Use Cases

### Real-Time Trading (Background Processes)
```bash
# Conservative setup - small positions (background process)
./start-real-time-trading start --symbols SPY QQQ --max-position 250
# Monitor with: BashOutput bash_X

# Active trading - momentum stocks (background process)  
./start-real-time-trading start --symbols TSLA NVDA AAPL MSFT --max-position 1000
# Monitor with: BashOutput bash_Y

# Monitor live performance via Claude Code background system
BashOutput bash_X  # Use actual bash_id returned by command

# Traditional monitoring (fallback)
./start-real-time-trading logs --log-lines 20
```

### Daily Trading
```
"Run my morning routine:
1. Check portfolio status
2. Analyze overnight news
3. Generate forecasts for watchlist
4. Identify best opportunities"
```

### Risk Management
```
"Assess my portfolio risk:
1. Calculate current VaR
2. Check correlations
3. Suggest hedges if needed"
```

### Strategy Development
```
"Help me develop a new strategy:
1. Backtest momentum on tech stocks
2. Optimize parameters
3. Compare to benchmark
4. Generate report"
```

## 🛠️ Available MCP Tools

### Neural Forecasting (6 tools)
- `neural_forecast` - AI price predictions
- `neural_train` - Train custom models
- `neural_evaluate` - Test model accuracy
- `neural_backtest` - Historical validation
- `neural_model_status` - Monitor models
- `neural_optimize` - Tune hyperparameters

### Trading Strategy (4 tools)
- `quick_analysis` - Technical indicators
- `simulate_trade` - Test trades
- `execute_trade` - Place orders
- `get_portfolio_status` - Portfolio analytics

### Advanced Analytics (7 tools)
- `run_backtest` - Test strategies
- `optimize_strategy` - Find best parameters
- `performance_report` - Detailed metrics
- `correlation_analysis` - Asset relationships
- `run_benchmark` - Performance testing
- `risk_analysis` - Portfolio risk
- `list_strategies` - Available strategies

### News & Sentiment (2 tools)
- `analyze_news` - AI sentiment analysis
- `get_news_sentiment` - Sentiment scores

### System (2 tools)
- `ping` - Check connection
- `get_strategy_info` - Strategy details

## 💡 Pro Tips

### 1. Use Natural Language
Instead of memorizing syntax, just describe what you want:
- ❌ `ai-news-trader:neural_forecast (MCP)(symbol: "AAPL", horizon: 24)`
- ✅ "Forecast Apple's price for tomorrow"

### 2. Request Parallel Execution
Speed up analysis by asking for simultaneous operations:
- "Analyze AAPL, MSFT, and GOOGL simultaneously"
- "Run forecasts for all timeframes in parallel"

### 3. Chain Operations
Connect tools with logical flow:
- "If Tesla sentiment is positive, then forecast price and simulate trade"
- "First check risk, then size position accordingly"

### 4. Use Memory
Persist important results:
- "Store this backtest as 'momentum_v2_results'"
- "Use the saved parameters to run another test"

### 5. Handle Errors Gracefully
Build in fallbacks:
- "Try neural forecast, if unavailable use technical analysis"
- "If connection fails, use cached data"

## 🚦 Getting Started Path

1. **Test Connection**
   ```
   "Ping the MCP server"
   ```

2. **Try Basic Analysis**
   ```
   "Quick analysis of AAPL"
   ```

3. **Generate First Forecast**
   ```
   "Neural forecast for SPY"
   ```

4. **Run a Backtest**
   ```
   "Backtest momentum strategy on TSLA"
   ```

5. **Build Your Workflow**
   ```
   "Create my daily trading routine"
   ```

## 📈 Example Conversation

```
You: "I want to find a good trade for tomorrow"

Claude Code: I'll help you find trading opportunities. Let me analyze the market...

[Executes multiple MCP tools in parallel]
- Scans top movers
- Analyzes news sentiment  
- Generates neural forecasts
- Checks technical indicators
- Assesses risk levels

Based on my analysis:
1. NVDA shows strong bullish sentiment (92%) with neural forecast predicting +3.2%
2. Technical indicators confirm uptrend
3. Risk metrics are acceptable
4. Suggested position size: 2.5% of portfolio

Would you like me to simulate this trade?
```

## 🆘 Troubleshooting

### MCP Server Not Responding
```
"Check MCP server status and restart if needed"
```

### Slow Performance
```
"Enable GPU acceleration for neural operations"
```

### Data Issues
```
"Validate data freshness and clear cache if stale"
```

## 📞 Support

- **Documentation**: You're here! 
- **Quick Help**: Ask Claude Code directly
- **Server Issues**: Check `.roo/mcp.json` configuration
- **Advanced Support**: See [Integration Guide](./mcp-integration-guide.md)

---

Ready to start? Just tell Claude Code what you want to analyze or trade!