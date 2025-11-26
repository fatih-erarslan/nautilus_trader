# Getting Started with Neural Trader

Welcome to Neural Trader! This section contains everything you need to get up and running quickly.

## 📖 Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Tutorials](#tutorials)
5. [Examples](#examples)
6. [Troubleshooting](#troubleshooting)

## 🚀 Installation & Setup

### Prerequisites
- Node.js >= 16.0.0
- npm >= 7.0.0
- Anthropic API key (for Claude integration)

### Quick Install

```bash
# Install neural-trader globally
npm install -g neural-trader

# Or use with npx (no installation needed)
npx neural-trader --help
```

### Detailed Guides
- [Complete Installation Guide](./guides/installation.md)
- [Quick Start Guide](./guides/quickstart.md)
- [Authentication Setup](./AUTHENTICATION.md)
- [Supabase Setup](./SUPABASE_SETUP_GUIDE.md)

## ⚡ Quick Start

### 1. Basic Analysis
```bash
# Analyze a stock
npx neural-trader analyze AAPL

# Generate forecast
npx neural-trader forecast BTC --horizon 24
```

### 2. Run a Strategy
```bash
# Backtest momentum strategy
npx neural-trader strategy --strategy momentum --symbol SPY --backtest --start 2020-01-01
```

### 3. Start MCP Server
```bash
# For AI assistant integration (Claude, Cursor, Copilot)
npx neural-trader mcp
```

See [Quick Start Guide](./guides/quickstart.md) for more examples.

## ⚙️ Configuration

### Environment Setup
Create `.env` file in your project root:

```env
ANTHROPIC_API_KEY=your_api_key_here
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
```

See [Configuration Guide](./configuration/system_config.md) for all options.

### Broker Setup
- [Alpaca Setup](../integrations/alpaca/CONFIGURE_ALPACA_MCP.md)
- [Supported APIs](./tutorials/101-introduction/12-supported-apis.md)

## 📚 Tutorials

Our tutorials are organized into learning paths:

### 101 - Introduction Series
1. [What is Neural Trader?](./tutorials/101-introduction/01-what-is-neural-trader.md)
2. [Installation & Setup](./tutorials/101-introduction/02-installation-setup.md)
3. [Claude Flow Basics](./tutorials/101-introduction/03-claude-flow-basics.md)
4. [Flow Nexus Setup](./tutorials/101-introduction/04-flow-nexus-setup.md)
5. [Claude Code UI](./tutorials/101-introduction/05-claude-code-ui.md)
6. [Basic Trading Strategies](./tutorials/101-introduction/06-basic-trading-strategies.md)
7. [Advanced Polymarket](./tutorials/101-introduction/07-advanced-polymarket.md)
8. [Sports Betting Syndicates](./tutorials/101-introduction/08-sports-betting-syndicates.md)
9. [Sandbox Workflows](./tutorials/101-introduction/09-sandbox-workflows.md)
10. [Neural Network Training](./tutorials/101-introduction/10-neural-network-training.md)
11. [Hello World Bot](./tutorials/101-introduction/11-hello-world-bot.md)
12. [Supported APIs](./tutorials/101-introduction/12-supported-apis.md)
13. [Optimization Strategies](./tutorials/101-introduction/13-optimization-strategies.md)

### Advanced Tutorials
- [GPU Optimization](./guides/VSCODE-GPU-SETUP.md)
- [Fly.io GPU Deployment](./guides/FLYIO_GPU_DEPLOYMENT.md)
- [Model Management](./guides/MODEL_MANAGEMENT_README.md)
- [Latency Optimization](./guides/LATENCY_OPTIMIZATION_GUIDE.md)

## 💡 Examples

Ready-to-run code examples in the [`examples/`](./examples/) directory:

- **Basic Forecasting:** [basic_forecasting.md](../getting-started/tutorials/basic_forecasting.md)
- **Advanced Features:** [advanced_features.md](../getting-started/tutorials/advanced_features.md)
- **GPU Optimization:** [gpu_optimization.md](../getting-started/tutorials/gpu_optimization.md)

## 🔧 Troubleshooting

Common issues and solutions:

- [Troubleshooting Guide](./TROUBLESHOOTING.md)
- [Environment Quick Reference](./guides/ENV_QUICK_REFERENCE.md)

### Quick Fixes

**Issue: Command not found**
```bash
npm install -g neural-trader
# Or use npx
npx neural-trader@latest --version
```

**Issue: API key errors**
```bash
# Check your .env file
cat .env | grep ANTHROPIC_API_KEY
```

**Issue: MCP server won't start**
```bash
# Check if port is in use
npx neural-trader mcp --port 3001
```

## 📖 Next Steps

After getting started:

1. Explore [API Reference](../api-reference/)
2. Learn about [Integrations](../integrations/)
3. Check out [Features](../features/)
4. Deploy to [Production](../deployment/)

## 🆘 Need Help?

- [Full Troubleshooting Guide](./TROUBLESHOOTING.md)
- [GitHub Issues](https://github.com/ruvnet/neural-trader/issues)
- [MCP Documentation](../api-reference/mcp/)

---

**Quick Links:**
- [← Back to Main Docs](../README.md)
- [API Reference →](../api-reference/)
- [Tutorials Directory →](./tutorials/)
