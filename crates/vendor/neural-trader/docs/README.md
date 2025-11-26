# Neural Trader Documentation

**Version:** 2.5.1
**Last Updated:** 2025-11-17

Welcome to the comprehensive Neural Trader documentation. This documentation will help you leverage all 178 functions and achieve production-grade trading systems.

---

## 📚 Quick Navigation

### Getting Started
- [Installation & Setup](./guides/getting-started.md)
- [Quick Start Guide](./guides/quick-start.md)
- [Basic Examples](./examples/)

### API Reference (178 Functions)
- [Classes (20)](./api/classes.md)
- [Market Data & Indicators (10)](./api/market-data.md)
- [Neural Networks (7)](./api/neural-networks.md)
- [Strategy & Backtest (14)](./api/strategy-backtest.md)
- [Trade Execution (8)](./api/trade-execution.md)
- [Portfolio Management (6)](./api/portfolio-management.md)
- [Risk Management (7)](./api/risk-management.md)
- [E2B Cloud Execution (13)](./api/e2b-cloud.md)
- [Sports Betting & Predictions (25)](./api/sports-betting.md)
- [Syndicate Management (18)](./api/syndicate-management.md)
- [News & Sentiment Analysis (9)](./api/news-sentiment.md)
- [Swarm Coordination (6)](./api/swarm-coordination.md)
- [Performance & Analytics (7)](./api/performance-analytics.md)
- [Data Science - DTW (5)](./api/dtw-data-science.md)
- [System Utilities (4)](./api/system-utilities.md)

### Integration Guides
- [Backtesting Guide](./guides/backtesting-guide.md)
- [Live Trading Guide](./guides/live-trading-guide.md)
- [Risk Management Guide](./guides/risk-management-guide.md)
- [Syndicate Setup](./guides/syndicate-setup.md)
- [Sports Betting Guide](./guides/sports-betting-guide.md)
- [Neural Networks Guide](./guides/neural-networks-guide.md)
- [E2B Deployment Guide](./guides/e2b-deployment-guide.md)
- [MCP Integration Guide](./guides/mcp-integration-guide.md)

---

## 🎯 Quick Start

\`\`\`javascript
const nt = require('neural-trader');

// 1. Fetch market data
const data = await nt.fetchMarketData('AAPL', '2024-01-01', '2024-12-31', 'alpaca');

// 2. Run backtest
const result = await nt.backtestStrategy('momentum', {
  symbol: 'AAPL',
  startDate: '2024-01-01',
  endDate: '2024-12-31',
  initialCapital: 10000
});
\`\`\`

---

## 📄 License

MIT OR Apache-2.0
