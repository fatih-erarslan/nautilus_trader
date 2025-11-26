# Neural Trader Packages Documentation

Complete documentation for all @neural-trader npm packages.

## 📦 Package Overview

Neural Trader is built as a modular monorepo with 17 specialized npm packages:

```
@neural-trader/
├── neural-trader          # Meta package (all features)
├── core                   # Type definitions
├── strategies             # Trading strategies
├── neural                 # Neural networks
├── portfolio              # Portfolio management
├── risk                   # Risk management
├── backtesting            # Backtesting engine
├── execution              # Order execution
├── features               # Technical indicators
├── market-data            # Market data feeds
├── brokers                # Broker integrations
├── mcp                    # MCP server
├── mcp-protocol           # MCP protocol types
├── sports-betting         # Sports betting
├── prediction-markets     # Prediction markets
├── syndicate              # Syndicate management
└── benchoptimizer         # Performance benchmarking
```

## 📚 Documentation Structure

### 📦 Package Documentation
Individual package API references and usage guides.

**Location:** [`/packages/`](./packages/)

Complete API documentation for all 17 packages with code examples, features, and status.

### 📖 Guides
Implementation guides and best practices.

**Location:** [`/guides/`](./guides/)

- [Getting Started with Packages](./guides/getting-started.md)
- Package selection guide
- Integration patterns
- Testing packages
- Publishing workflow

### 🏗️ Architecture
Modular architecture and design decisions.

**Location:** [`/architecture/`](./architecture/)

- Modular architecture overview
- Package dependencies
- NAPI bindings design
- Build system

### 💻 Development
Development workflows, build system, and publishing.

**Location:** [`/development/`](./development/)

- [Development Guide](./development/README.md)
- Build and publishing docs
- Test results and summaries
- Verification reports

### 🧪 Tests
Test results and coverage reports.

**Location:** [`/tests/`](./tests/)

- Comprehensive test reports
- Package test results
- E2B swarm tests
- MCP tool catalog

## 🚀 Quick Start

### Installing Packages

**Option 1: Meta Package (All Features)**
```bash
npm install neural-trader
```

**Option 2: Individual Packages (Minimal)**
```bash
npm install @neural-trader/core @neural-trader/strategies
```

### Using Packages

```typescript
// Import from meta package
import { NeuralTrader } from 'neural-trader';

// Or import from individual packages
import { Strategy } from '@neural-trader/strategies';
import { NeuralModel } from '@neural-trader/neural';
import { RiskManager } from '@neural-trader/risk';
```

## 📊 Package Status

| Package | Version | Status | Dependencies |
|---------|---------|--------|--------------|
| neural-trader | 1.0.12 | ✅ Stable | All packages |
| @neural-trader/core | 1.0.0 | ✅ Stable | 0 |
| @neural-trader/strategies | 1.0.0 | ✅ Stable | 1 |
| @neural-trader/neural | 1.0.0 | ✅ Stable | 1 |
| @neural-trader/portfolio | 1.0.0 | ✅ Stable | 1 |
| @neural-trader/risk | 1.0.0 | ✅ Stable | 1 |
| @neural-trader/backtesting | 1.0.0 | ✅ Stable | 1 |
| @neural-trader/execution | 1.0.0 | ⚠️ Fix Needed | 1 |
| @neural-trader/features | 1.0.0 | ⚠️ Fix Needed | 1 |
| @neural-trader/market-data | 1.0.0 | ✅ Stable | 0 |
| @neural-trader/brokers | 1.0.0 | ✅ Stable | 0 |
| @neural-trader/news-trading | 1.0.0 | ⚠️ Placeholder | 7 |
| @neural-trader/sports-betting | 1.0.0 | ⚠️ Partial | 1 |
| @neural-trader/prediction-markets | 1.0.0 | ❌ Empty | 1 |
| @neural-trader/syndicate | 1.0.0 | ✅ Stable | 4 |
| @neural-trader/mcp | 1.0.0 | ✅ Stable | 2 |
| @neural-trader/mcp-protocol | 1.0.0 | ✅ Stable | 1 |
| @neural-trader/benchoptimizer | 1.0.0 | ✅ Stable | 12 |

**Legend:**
- ✅ Stable: Production-ready
- ⚠️ Fix Needed: Known issues (see GitHub issues)
- ⚠️ Partial: Incomplete implementation
- ⚠️ Placeholder: Empty/minimal implementation
- ❌ Empty: No implementation

## 🐛 Known Issues

See [GitHub Issues](https://github.com/ruvnet/neural-trader/issues) for full list:

**Critical (P0):**
- [#69](https://github.com/ruvnet/neural-trader/issues/69): Hardcoded native binding paths in execution/features packages

**High (P1):**
- [#70](https://github.com/ruvnet/neural-trader/issues/70): RSI calculation returns NaN values

**Medium (P2):**
- [#71](https://github.com/ruvnet/neural-trader/issues/71): Remove unnecessary dependencies from news-trading
- [#72](https://github.com/ruvnet/neural-trader/issues/72): Implement sports-betting and prediction-markets
- [#73](https://github.com/ruvnet/neural-trader/issues/73): Add test suites across packages

## 📖 Common Use Cases

### Trading Bot Development
```bash
npm install @neural-trader/strategies @neural-trader/execution @neural-trader/risk
```

### Neural Forecasting
```bash
npm install @neural-trader/neural @neural-trader/market-data
```

### Backtesting Only
```bash
npm install @neural-trader/backtesting @neural-trader/strategies
```

### Full Platform
```bash
npm install neural-trader
```

### Sports Betting Syndicate
```bash
npm install @neural-trader/syndicate @neural-trader/sports-betting
```

## 🔗 Related Documentation

- [Main Documentation](../../../../docs/)
- [Getting Started](../../../../docs/getting-started/)
- [API Reference](../../../../docs/api-reference/)
- [Architecture](../../../../docs/architecture/)
- [Package Test Reports](../../../tests/sub-package-tests/)

## 📦 Publishing

All packages are published to npm under the `@neural-trader` scope:

```bash
# View package on npm
npm view @neural-trader/strategies

# Install specific version
npm install @neural-trader/strategies@1.0.0
```

## 🆘 Support

- [GitHub Issues](https://github.com/ruvnet/neural-trader/issues)
- [Package Test Reports](../../../tests/sub-package-tests/)
- [Main Documentation](../../../../docs/)

---

**Quick Navigation:**
- [Package API Reference →](./packages/)
- [Getting Started Guide →](./guides/getting-started.md)
- [Development Guide →](./development/)
- [Test Results →](./tests/)

**Last Updated:** 2025-11-14
