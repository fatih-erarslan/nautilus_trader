#!/bin/bash
# AI News Trading Platform - Parallel Agent Execution
# Generated: 2025-06-28T17:26:34.364225

echo '🚀 Starting AI News Trading Platform Demo Swarm'
echo '================================================'
echo ''
echo '📊 Configuration:'
echo '  - Symbols: AAPL, NVDA, TSLA, GOOGL, MSFT'
echo '  - Strategies: momentum_trading_optimized, swing_trading_optimized'
echo '  - Markets: crypto_btc_100k, crypto_eth_5000'
echo '  - Timeframe: 2025-05-29 to 2025-06-28'
echo ''
echo '🤖 Launching 5 Parallel Agents...'
echo ''

# Execute agents in parallel using batchtool
batchtool \
  --parallel 5 \
  --timeout 300 \
  --output-format json \
  --agent1 'python execute_agent.py --agent-id market_analyst' \
  --agent2 'python execute_agent.py --agent-id news_analyst' \
  --agent3 'python execute_agent.py --agent-id strategy_optimizer' \
  --agent4 'python execute_agent.py --agent-id risk_manager' \
  --agent5 'python execute_agent.py --agent-id trader' \
  --verbose