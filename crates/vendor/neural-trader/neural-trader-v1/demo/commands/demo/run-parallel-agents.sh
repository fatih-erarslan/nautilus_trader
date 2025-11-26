#!/bin/bash
# Claude Flow Command: Demo - Run Parallel Trading Agents

echo "🚀 Claude Flow: Parallel Trading Agent Demo"
echo "=========================================="
echo ""
echo "Launching 5 specialized AI trading agents in parallel..."
echo ""

# Set up environment
export DEMO_MODE=true
export USE_GPU=true
export PYTHONPATH=/workspaces/ai-news-trader:$PYTHONPATH

# Configuration
SYMBOLS="AAPL,NVDA,TSLA,GOOGL,MSFT"
STRATEGIES="momentum_trading_optimized,swing_trading_optimized"
TIMEFRAME="30d"

echo "📊 Configuration:"
echo "  - Symbols: $SYMBOLS"
echo "  - Strategies: $STRATEGIES"
echo "  - Timeframe: $TIMEFRAME"
echo "  - GPU: Enabled"
echo ""

# Launch agents using Claude Code Task tool prompts
cat << 'EOF'
🤖 Agent Tasks:

1. Market Analysis Agent - Analyzing market conditions with neural forecasting
2. News Sentiment Agent - Aggregating and analyzing news from multiple sources
3. Strategy Optimizer - Comparing and optimizing trading strategies
4. Risk Manager - Calculating portfolio risk with Monte Carlo simulations
5. Trading Executor - Simulating trades and analyzing prediction markets

To execute these agents in Claude Code, use the Task tool with the following prompts:

---

### Agent 1: Market Analysis
Task Description: Market Analysis Agent
Prompt: Analyze AAPL, NVDA, and TSLA using mcp__ai-news-trader__quick_analysis with use_gpu: true, then generate 7-day forecast for NVDA using mcp__ai-news-trader__neural_forecast with horizon: 7, confidence_level: 0.95, use_gpu: true. Finally check system metrics with mcp__ai-news-trader__get_system_metrics.

### Agent 2: News Sentiment
Task Description: News Sentiment Analyst
Prompt: Start news collection using mcp__ai-news-trader__control_news_collection with action: "start", symbols: ["AAPL", "NVDA", "TSLA", "GOOGL", "MSFT"]. Analyze TSLA sentiment with mcp__ai-news-trader__analyze_news using lookback_hours: 48, sentiment_model: "enhanced". Get trends with mcp__ai-news-trader__get_news_trends.

### Agent 3: Strategy Optimization
Task Description: Strategy Optimization Agent
Prompt: List strategies with mcp__ai-news-trader__list_strategies, compare them using mcp__ai-news-trader__get_strategy_comparison with strategies: ["momentum_trading_optimized", "swing_trading_optimized"], then run backtest with mcp__ai-news-trader__run_backtest for NVDA.

### Agent 4: Risk Management
Task Description: Risk Management Agent
Prompt: Get portfolio status with mcp__ai-news-trader__get_portfolio_status, generate correlation matrix using mcp__ai-news-trader__cross_asset_correlation_matrix for ["AAPL", "NVDA", "TSLA", "GOOGL", "MSFT"], then run risk analysis with mcp__ai-news-trader__risk_analysis.

### Agent 5: Trading Execution
Task Description: Trading Execution Agent
Prompt: Simulate NVDA trade with mcp__ai-news-trader__simulate_trade, get prediction markets using mcp__ai-news-trader__get_prediction_markets_tool, calculate expected value with mcp__ai-news-trader__calculate_expected_value_tool, then generate performance report with mcp__ai-news-trader__performance_report.

EOF

echo ""
echo "✅ Demo agents ready to launch in Claude Code!"
echo "📝 Copy the agent prompts above and use them with the Task tool"
echo ""

# Alternative: Run local simulation
read -p "Run local simulation? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running local simulation..."
    cd /workspaces/ai-news-trader/demo/scripts
    python execute_agent.py --agent-id market_analyst &
    python execute_agent.py --agent-id news_analyst &
    python execute_agent.py --agent-id strategy_optimizer &
    python execute_agent.py --agent-id risk_manager &
    python execute_agent.py --agent-id trader &
    wait
    echo "✅ Local simulation complete!"
fi