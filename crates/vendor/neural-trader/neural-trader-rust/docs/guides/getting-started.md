# Getting Started with Neural Trader

Complete guide to installing, configuring, and running Neural Trader.

## 📦 Installation

### Method 1: NPM (Recommended)

```bash
# Install globally
npm install -g neural-trader

# Or use via npx (no installation)
npx neural-trader --help
```

### Method 2: Cargo

```bash
# Install from crates.io
cargo install neural-trader

# Verify installation
neural-trader --version
```

### Method 3: From Source

```bash
# Clone repository
git clone https://github.com/ruvnet/neural-trader
cd neural-trader/neural-trader-rust

# Build release binary
cargo build --release

# Binary located at: target/release/neural-trader
```

## 🔧 Prerequisites

### Required
- **Operating System**: Linux, macOS, or Windows
- **Node.js**: 18.0.0 or higher (for NPM installation)
- **Rust**: 1.70.0 or higher (for Cargo installation)

### Optional
- **CUDA Toolkit**: For GPU acceleration
- **Docker**: For containerized deployment
- **PostgreSQL**: For persistent storage

## ⚙️ Configuration

### 1. Environment Variables

Create `.env` file in your project directory:

```env
# Alpaca API (Paper Trading - Safe for Testing)
ALPACA_API_KEY=PKXXXXXXXXXXXXXXXXXX
ALPACA_SECRET_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Alpaca API (Live Trading - REAL MONEY!)
# ALPACA_BASE_URL=https://api.alpaca.markets

# AgentDB Configuration (Optional)
AGENTDB_URL=http://localhost:8080
AGENTDB_COLLECTION=neural-trader

# Logging Configuration
RUST_LOG=info,neural_trader=debug
RUST_BACKTRACE=1

# Performance Tuning
TOKIO_WORKER_THREADS=4
RAYON_NUM_THREADS=8
```

### 2. Get API Keys

#### Alpaca (Stock Trading)
1. Visit [https://alpaca.markets](https://alpaca.markets)
2. Create account (free)
3. Navigate to "API Keys" in dashboard
4. Generate new API key
5. Copy API Key and Secret Key to `.env`

⚠️ **Important**: Always start with Paper Trading (paper-api.alpaca.markets) to test strategies safely!

### 3. Verify Configuration

```bash
# Test API connection
npx neural-trader test-connection

# Expected output:
# ✓ Alpaca API: Connected
# ✓ Market Status: Open
# ✓ Account Status: Active
```

## 🚀 Quick Start Tutorial

### Example 1: Fetch Market Data

```bash
# Fetch historical data
npx neural-trader market-data \
  --symbol AAPL \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --output data/aapl.csv

# Output:
# ✓ Downloaded 252 daily bars
# ✓ Saved to data/aapl.csv
```

### Example 2: Run a Backtest

```bash
# Backtest pairs trading strategy
npx neural-trader backtest \
  --strategy pairs \
  --symbols AAPL MSFT \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --initial-capital 100000

# Output:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BACKTEST RESULTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Strategy: Pairs Trading
# Period: 2024-01-01 to 2024-12-31
# Initial Capital: $100,000.00
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Total Return: +18.3%
# Sharpe Ratio: 1.85
# Max Drawdown: -8.2%
# Win Rate: 64.5%
# Total Trades: 127
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Example 3: Train Neural Model

```bash
# Train N-HiTS forecasting model
npx neural-trader neural train \
  --model nhits \
  --data data/aapl.csv \
  --horizon 5 \
  --epochs 100 \
  --output models/aapl_nhits.bin

# Output:
# Epoch 1/100: Loss = 0.0234
# Epoch 2/100: Loss = 0.0198
# ...
# Epoch 100/100: Loss = 0.0045
# ✓ Model saved to models/aapl_nhits.bin
```

### Example 4: Calculate Risk Metrics

```bash
# Calculate Value at Risk (VaR)
npx neural-trader risk var \
  --portfolio portfolio.json \
  --confidence 0.95 \
  --method monte-carlo \
  --simulations 10000

# Output:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RISK ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Method: Monte Carlo Simulation
# Confidence Level: 95%
# Simulations: 10,000
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VaR (1-day, 95%): $2,450
# CVaR (Expected Shortfall): $3,120
# Maximum Loss (99.9%): $8,900
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Example 5: Start MCP Server

```bash
# Start MCP server for Claude integration
npx neural-trader mcp start

# Output:
# ✓ MCP server listening on stdio
# ✓ 50+ tools registered
# ✓ Ready for Claude integration
```

## 📊 Portfolio Configuration

Create `portfolio.json`:

```json
{
  "name": "My Portfolio",
  "initial_capital": 100000.0,
  "positions": [
    {
      "symbol": "AAPL",
      "quantity": 50,
      "entry_price": 180.50
    },
    {
      "symbol": "MSFT",
      "quantity": 30,
      "entry_price": 380.25
    },
    {
      "symbol": "GOOGL",
      "quantity": 20,
      "entry_price": 140.75
    }
  ],
  "risk_limits": {
    "max_position_size": 0.25,
    "max_portfolio_var": 0.05,
    "stop_loss_percent": 0.08
  }
}
```

## 🎯 Strategy Configuration

Create `strategy-config.toml`:

```toml
[strategy]
name = "pairs_trading"
type = "statistical_arbitrage"

[strategy.parameters]
symbols = ["AAPL", "MSFT"]
lookback_period = 20
entry_threshold = 2.0
exit_threshold = 0.5
position_size = 0.1

[risk]
max_drawdown = 0.15
stop_loss = 0.08
take_profit = 0.12

[execution]
broker = "alpaca"
order_type = "limit"
timeout_seconds = 30
```

## 🧪 Testing Your Setup

```bash
# Run system health check
npx neural-trader health-check

# Expected output:
# ✓ Configuration valid
# ✓ API credentials valid
# ✓ Market data accessible
# ✓ Order execution enabled
# ✓ Risk limits configured
# ✓ All systems operational
```

## 🐛 Troubleshooting

### Issue: "Command not found"

```bash
# Ensure neural-trader is installed
npm list -g neural-trader

# If not found, reinstall
npm install -g neural-trader
```

### Issue: "API authentication failed"

```bash
# Verify .env file exists
cat .env

# Test API connection
npx neural-trader test-connection --verbose
```

### Issue: "Module not found"

```bash
# Reinstall dependencies
npm install

# Rebuild native bindings
npm run build:napi
```

## 📚 Next Steps

1. **Learn Strategies**: Read [Strategy Development Guide](./strategy-development.md)
2. **Integrate with Claude**: See [MCP Integration](./mcp-integration.md)
3. **Deploy to Production**: Follow [Deployment Guide](./deployment.md)
4. **Optimize Performance**: Check [Performance Tuning](./performance.md)

## 🎓 Learning Path

### Beginner
1. Run example backtests
2. Understand strategy results
3. Experiment with parameters
4. Calculate basic risk metrics

### Intermediate
1. Create custom strategies
2. Train neural models
3. Optimize portfolio allocation
4. Implement risk management

### Advanced
1. Build multi-strategy systems
2. Deploy to production
3. Integrate with AgentDB
4. Contribute to codebase

## 💡 Tips for Success

- **Always start with paper trading** - Test strategies safely
- **Use realistic parameters** - Don't over-optimize
- **Monitor risk closely** - Use stop-losses and position limits
- **Backtest thoroughly** - Validate strategies on historical data
- **Start small** - Scale up after proving profitability
- **Keep learning** - Markets evolve, strategies must too

## 🤝 Getting Help

- **Documentation**: Read the full docs
- **Examples**: Check `examples/` directory
- **Issues**: Report bugs on GitHub
- **Discussions**: Ask questions in Discussions
- **Discord**: Join community for real-time help

---

Ready to trade? Let's build wealth with code! 🚀
