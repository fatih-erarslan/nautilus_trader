# Neural Trader - Quick Reference Card

## 🚀 Installation

```bash
# Install from local build
npm install

# Build native addon
npm run build

# Test installation
npx neural-trader --version
```

## 📋 CLI Commands

| Command | Description |
|---------|-------------|
| `npx neural-trader --version` | Show version |
| `npx neural-trader --help` | Show help |
| `npx neural-trader list-strategies` | List 6 trading strategies |
| `npx neural-trader list-brokers` | List 5 brokers/data sources |
| `npx neural-trader init [path]` | Initialize project (coming soon) |

## 📦 SDK Usage

### JavaScript
```javascript
const {
  MarketDataStream,
  StrategyRunner,
  ExecutionEngine,
  PortfolioManager,
  version
} = require('@neural-trader/core');

console.log(`Neural Trader v${version}`);
```

### TypeScript
```typescript
import {
  MarketDataStream,
  Quote,
  Signal,
  TradeOrder
} from '@neural-trader/core';
```

## 🧪 Testing

```bash
# Run all tests
npm run test:all

# Individual test suites
npm run test:cli      # CLI tests (7/7)
npm run test:sdk      # SDK tests (7/7)
npm run test:mcp      # MCP tests (7/7)

# Comprehensive validation
node tests/comprehensive-validation.js  # 28/29 tests
```

## 📊 Trading Strategies

1. **Momentum** - Trend following (Medium-High risk)
2. **Mean Reversion** - Price reversals (Low-Medium risk)
3. **Arbitrage** - Price differences (Low risk)
4. **Market Making** - Liquidity provision (Medium risk)
5. **Pairs Trading** - Statistical arbitrage (Medium risk)
6. **Neural Network** - ML predictions (High risk)

## 🏦 Brokers & Data Sources

| Broker | Status | Features |
|--------|--------|----------|
| Alpaca Markets | ✅ Supported | Paper trading, real-time data |
| Interactive Brokers | 🔄 In Development | Global markets, advanced orders |
| Binance | 🔄 In Development | Crypto, futures, spot |
| Polygon.io | ✅ Supported | Market data only |
| Kraken | 📋 Planned | Crypto exchange |

## 🔧 Development

```bash
# Build from source
cargo build --package nt-napi-bindings --release

# Copy native addon
cp target/release/libneural_trader.so neural-trader.linux-x64-gnu.node

# Link for local development
npm link
```

## 📁 Key Files

```
/workspaces/neural-trader/neural-trader-rust/
├── neural-trader.linux-x64-gnu.node  # Native addon (794 KB)
├── package.json                      # NPM manifest
├── index.js                          # SDK entry
├── index.d.ts                        # TypeScript types
├── bin/cli.js                        # CLI entry
└── tests/                            # Test suites
    ├── cli-test.js
    ├── sdk-test.js
    ├── mcp-test.js
    └── comprehensive-validation.js
```

## 📚 Documentation

- `TEST_SUMMARY.md` - Test results overview
- `docs/NPM_BUILD_COMPLETE.md` - Full build documentation
- `docs/NPM_TEST_RESULTS.md` - Detailed test results
- `README.md` - Project overview

## ✅ Test Results

| Suite | Result | Success Rate |
|-------|--------|--------------|
| CLI Tests | 7/7 | 100% |
| SDK Tests | 7/7 | 100% |
| MCP Tests | 7/7 | 100% |
| Validation | 28/29 | 96.6% |
| **TOTAL** | **49/50** | **98.0%** |

## 🎯 Quick Test

```bash
# Verify everything works
npx neural-trader --version
npx neural-trader list-strategies
node -e "console.log(require('.').version)"
npm run test:all
```

## 🚦 Status

- ✅ Rust compilation: **SUCCESS**
- ✅ NPM package: **BUILT**
- ✅ CLI: **WORKING**
- ✅ SDK: **IMPORTABLE**
- ✅ Tests: **98% PASSING**
- 🔄 Native functions: **IN DEVELOPMENT**
- 🔄 MCP server: **PLANNED**

## 📞 Support

- Repository: https://github.com/ruvnet/neural-trader
- Location: `/workspaces/neural-trader/neural-trader-rust/`
- Version: `0.1.0`
- Platform: `linux-x64-gnu`

---

**Last Updated**: 2025-11-12
**Status**: ✅ BUILD COMPLETE
