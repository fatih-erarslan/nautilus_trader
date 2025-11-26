# Neural Trader - Rust Documentation

High-performance neural trading system built in Rust with GPU acceleration, real-time execution, and comprehensive backtesting capabilities.

## 📚 Documentation Index

### 🚀 User Guides
- [Getting Started](./guides/getting-started.md) - Installation, setup, and first steps
- [API Reference](./guides/api-reference.md) - Complete API documentation
- [Migration Guide](./guides/migration-guide.md) - Migrating from Python to Rust

### 🏗️ Architecture & Design
- [Integration Architecture](./architecture/INTEGRATION_ARCHITECTURE.md) - System integration patterns
- [Feature Gate Patterns](./architecture/FEATURE_GATE_PATTERNS.md) - Conditional compilation

### 💻 Development Guides
- [Development Guide](./development/DEVELOPMENT.md) - Development workflows
- [MCP Tools Implementation](./development/MCP_TOOLS_IMPLEMENTATION.md) - MCP protocol
- [NAPI CLI Implementation](./development/NAPI_CLI_IMPLEMENTATION.md) - Node.js bindings
- [Execution System](./development/EXECUTION_CRATE_FIXES.md) - Order execution
- [Risk Management](./development/EXECUTION_RISK_IMPLEMENTATION.md) - Risk controls
- [WebSocket Streaming](./development/polygon-websocket-implementation.md) - Real-time data

### 🧠 Neural Networks
- [Neural Models](./neural/) - AI/ML model documentation (50+ docs)
  - Training guides, benchmarks, optimization, AgentDB integration

### 🎯 Trading Strategies
- [Strategy Implementation](./strategies/STRATEGIES_IMPLEMENTATION_SUMMARY.md)
- [Strategy Status](./strategies/STRATEGIES_FINAL_STATUS.md)
- [Integration Status](./strategies/STRATEGY_INTEGRATION_STATUS.md)

### 📊 Features & Parity
- [Feature Audit](./features/FEATURE_AUDIT_SUMMARY.md) - Complete feature inventory
- [Python-Rust Parity](./features/PYTHON_RUST_FEATURE_PARITY.md) - Parity tracking
- [Parity Dashboard](./features/PARITY_DASHBOARD.md) - Implementation status

### 🔬 Testing & Validation
- **Testing:**
  - [Testing Guide](./testing/TESTING_GUIDE.md)
  - [Test Summary](./testing/TEST_SUMMARY.md)
  - [Test Infrastructure](./testing/TEST_INFRASTRUCTURE_COMPLETE.md)
- **Validation:**
  - [Validation Quick Start](./validation/VALIDATION_QUICKSTART.md)
  - [Validation Report](./validation/VALIDATION_REPORT.md)
  - [Final Validation](./validation/FINAL_VALIDATION_REPORT.md)

### 🏁 Project Status
- [Completion Summary](./completion/COMPLETION_SUMMARY.md)
- [Implementation Complete](./completion/IMPLEMENTATION_COMPLETE.md)
- [Phase 4 Summary](./completion/PHASE_4_COMPLETION_SUMMARY.md)

### 🔧 Build & Distribution
- [NPM Build](./build/NPM_BUILD_COMPLETE.md)
- [Publishing Guide](./build/NPM_PUBLISHING.md)
- [Setup Summary](./build/NPM_SETUP_SUMMARY.md)

### 🚢 Releases
- [Release Guide](./releases/RELEASE.md)
- [Security Policy](./releases/SECURITY.md)
- [Production Readiness](./releases/PRODUCTION_READINESS.md)

### 📋 Reference
- [Quick Reference](./reference/QUICK_REFERENCE.md) - Commands cheatsheet
- [Performance](./reference/performance.md) - Benchmarks
- [Compilation Fixes](./reference/COMPILATION_FIX_REPORT.md)

### 📈 Trading Results
- [Alpaca Paper Trading](./trading/ALPACA_PAPER_TRADING_RESULTS.md)

### 📝 Reports & Research
- [Agent Reports](./reports/) - Development status reports
- [Research](./research/) - Technical research and decisions

## 🚀 Quick Start

### Installation

```bash
# NPM (recommended)
npx neural-trader --help

# Cargo
cargo install neural-trader

# From source
git clone https://github.com/ruvnet/neural-trader
cd neural-trader/neural-trader-rust
cargo build --release
```

### Basic Usage

```bash
# Start MCP server
npx neural-trader mcp start

# Run backtest
npx neural-trader backtest \
  --strategy pairs \
  --symbol AAPL \
  --start 2024-01-01 \
  --end 2024-12-31

# Train neural model
npx neural-trader neural train \
  --model nhits \
  --data ./data/prices.csv \
  --output ./models/nhits.bin

# Risk analysis
npx neural-trader risk \
  --portfolio ./portfolio.json \
  --method monte-carlo \
  --confidence 0.95
```

## 🏗️ Architecture

Neural Trader is organized into 15 specialized crates:

### Core Crates
- **core** - Common types, traits, and utilities
- **market-data** - Real-time and historical data feeds
- **features** - Feature engineering and technical indicators
- **strategies** - Trading strategy implementations

### Execution & Portfolio
- **execution** - Order routing and broker integration
- **portfolio** - Position tracking and P&L calculation
- **risk** - VaR, position sizing, and risk limits
- **backtesting** - Historical strategy simulation

### Advanced Features
- **neural** - Neural network models for forecasting
- **agentdb-client** - AgentDB integration for learning
- **streaming** - Real-time data streaming
- **governance** - Multi-signature and role-based access

### Interface Crates
- **cli** - Command-line interface
- **napi-bindings** - Node.js bindings
- **utils** - Shared utilities

## 🔧 Configuration

Create `.env` file:

```env
# Broker API Keys
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# AgentDB (optional)
AGENTDB_URL=http://localhost:8080
AGENTDB_COLLECTION=neural-trader

# Logging
RUST_LOG=info
RUST_BACKTRACE=1
```

## 📊 Performance

Rust implementation provides:
- **3-10x faster** than Python baseline
- **Sub-millisecond** order execution latency
- **90%+ code coverage** with comprehensive tests
- **GPU acceleration** for neural computations
- **Zero-copy** data processing with Polars

See [Performance Benchmarks](./performance.md) for detailed metrics.

## 🧪 Testing

```bash
# Run all tests
cargo test --workspace

# Run with coverage
cargo tarpaulin --workspace --out Html

# Run benchmarks
cargo bench --workspace

# Integration tests
cargo test --test '*' --all-features
```

## 🤝 Integration Examples

### Node.js

```javascript
const { backtest, runStrategy } = require('neural-trader');

const results = await backtest('pairs', {
  symbols: ['AAPL', 'MSFT'],
  lookback: 20,
  entryThreshold: 2.0
}, '2024-01-01', '2024-12-31');

console.log(`Total Return: ${results.totalReturn}%`);
console.log(`Sharpe Ratio: ${results.sharpeRatio}`);
```

### Python (via subprocess)

```python
import subprocess
import json

result = subprocess.run([
    'npx', 'neural-trader', 'backtest',
    '--strategy', 'pairs',
    '--symbol', 'AAPL',
    '--json'
], capture_output=True, text=True)

data = json.loads(result.stdout)
print(f"Total Return: {data['total_return']}%")
```

### Rust

```rust
use nt_strategies::{PairsStrategy, Strategy};
use nt_backtesting::Backtester;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let strategy = PairsStrategy::new(vec!["AAPL", "MSFT"], 20, 2.0);
    let backtester = Backtester::new(strategy);

    let results = backtester.run("2024-01-01", "2024-12-31").await?;
    println!("Sharpe Ratio: {}", results.sharpe_ratio);

    Ok(())
}
```

## 🔌 MCP Tools

Neural Trader provides 50+ MCP tools for:
- Market data streaming
- Strategy backtesting
- Neural model training
- Portfolio optimization
- Risk management
- Sports betting integration

See [MCP Integration Guide](./mcp-integration.md) for details.

## 📈 Supported Strategies

- **Pairs Trading** - Statistical arbitrage
- **Mean Reversion** - Reversion to mean
- **Momentum** - Trend following
- **Market Making** - Liquidity provision
- **Neural Sentiment** - AI-driven sentiment analysis
- **Multi-Strategy** - Portfolio of strategies

Custom strategies can be implemented via the `Strategy` trait.

## 🛠️ Development

```bash
# Format code
cargo fmt --all

# Lint
cargo clippy --workspace -- -D warnings

# Build docs
cargo doc --workspace --no-deps --open

# Watch mode (requires cargo-watch)
cargo watch -x test
```

## 🐛 Troubleshooting

### Common Issues

**"No native binding found"**
```bash
npm run build:napi
```

**"Alpaca API authentication failed"**
- Check `.env` file has correct API keys
- Verify API key permissions in Alpaca dashboard

**"CUDA not found"**
- Install CUDA toolkit for GPU acceleration
- Build with: `cargo build --features gpu`

See [Troubleshooting Guide](./troubleshooting.md) for more solutions.

## 📝 License

Dual-licensed under MIT OR Apache-2.0

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## 📞 Support

- GitHub Issues: https://github.com/ruvnet/neural-trader/issues
- Documentation: https://docs.rs/neural-trader
- Discussions: https://github.com/ruvnet/neural-trader/discussions

---

Built with ❤️ using Rust for maximum performance and safety.
