# AI News Trading Platform - Demo Suite

## 📁 Demo Structure

```
demo/
├── README.md           # This file
├── docs/              # Main documentation
│   ├── DEMO_INDEX.md           # Master index of all demos
│   ├── DEMO_SWARM_GUIDE.md     # Comprehensive swarm guide
│   ├── INTERACTIVE_DEMO.md      # Interactive demo instructions
│   └── PARALLEL_AGENT_EXECUTION.md  # Parallel execution guide
├── guides/            # Individual feature demos
│   ├── market_analysis_demo.md    # Market analysis walkthrough
│   ├── news_analysis_demo.md      # News sentiment demo
│   ├── strategy_optimization_demo.md  # Strategy optimization
│   ├── risk_management_demo.md    # Risk analysis demo
│   └── trading_execution_demo.md  # Trading workflow
└── scripts/           # Executable demo scripts
    ├── demo_trading_swarm.py      # Main demo generator
    ├── demo_parallel_swarm.py     # Parallel swarm creator
    ├── run_trading_demo_swarm.py  # Trading demo runner
    ├── execute_agent.py           # Individual agent executor
    └── run_parallel_demo.sh       # Batch execution script
```

## 🚀 Quick Start

### Option 1: Run Complete Parallel Demo
Navigate to the parallel execution guide:
```bash
cd demo/docs
# Open PARALLEL_AGENT_EXECUTION.md and follow instructions
```

### Option 2: Run Individual Feature Demos
Explore specific capabilities:
```bash
cd demo/guides
# Choose any demo file:
# - market_analysis_demo.md
# - news_analysis_demo.md
# - strategy_optimization_demo.md
# - risk_management_demo.md
# - trading_execution_demo.md
```

### Option 3: Execute Demo Scripts
Run the automated demo scripts:
```bash
cd demo/scripts

# Generate demo files
python demo_trading_swarm.py

# Run individual agent
python execute_agent.py --agent-id market_analyst

# Run all agents in parallel
./run_parallel_demo.sh
```

## 📊 Available Demos

### 1. Market Analysis
- Real-time price analysis with GPU acceleration
- 7-day AI neural forecasts
- Technical indicators and trading signals
- System performance monitoring

### 2. News Sentiment
- Multi-source news aggregation
- AI sentiment analysis with FinBERT
- Trend analysis across timeframes
- High-impact news filtering

### 3. Strategy Optimization
- Strategy comparison and backtesting
- Parameter optimization with ML
- Adaptive strategy selection
- Performance attribution

### 4. Risk Management
- Portfolio correlation analysis
- Monte Carlo VaR simulations
- Optimal rebalancing calculations
- Strategy health monitoring

### 5. Trading Execution
- Trade simulation and execution
- Prediction market analysis
- Multi-asset batch trading
- Performance reporting

## 🛠️ MCP Tools Used

All demos utilize the 41 verified MCP tools with the prefix:
```
mcp__ai-news-trader__[tool_name]
```

### Tool Categories:
- System Tools (2)
- Trading Strategy (4)
- Portfolio Management (1)
- Neural Forecasting (6)
- Advanced Analytics (7)
- News & Sentiment (2)
- Prediction Markets (6)
- News Collection (4)
- Strategy Selection (4)
- Performance Monitoring (3)
- Multi-Asset Trading (3)

## 💡 Best Practices

1. **Start with DEMO_INDEX.md** in the docs folder for a complete overview
2. **Enable GPU acceleration** by adding `use_gpu: true` to tool parameters
3. **Run agents in parallel** for realistic trading scenarios
4. **Combine multiple tools** for comprehensive analysis
5. **Check system metrics** to monitor performance

## 📈 Expected Performance

- **Execution Time**: <5 seconds for all 5 agents
- **GPU Speedup**: 1000x for neural operations
- **Concurrent Capacity**: 200+ users
- **P95 Latency**: <1 second
- **Cache Hit Rate**: 95%+

## 🔗 Related Resources

- Main README: [/README.md](/README.md)
- MCP Documentation: [/CLAUDE.md](/CLAUDE.md)
- Implementation Status: [/IMPLEMENTATION_STATUS.md](/IMPLEMENTATION_STATUS.md)

---

Ready to explore? Start with [docs/DEMO_INDEX.md](docs/DEMO_INDEX.md) for the complete demo experience!