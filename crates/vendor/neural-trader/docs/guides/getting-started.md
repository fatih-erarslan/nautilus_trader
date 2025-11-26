# Getting Started with Neural Trader Backend

Complete beginner's guide to using the Neural Trader Backend API.

## Table of Contents

1. [Installation](#installation)
2. [First Steps](#first-steps)
3. [Basic Concepts](#basic-concepts)
4. [Your First Trade](#your-first-trade)
5. [Common Patterns](#common-patterns)
6. [Troubleshooting](#troubleshooting)
7. [Next Steps](#next-steps)

---

## Installation

### Prerequisites

- Node.js 16+ or 18+
- npm or yarn
- 2GB+ RAM
- (Optional) CUDA-capable GPU for acceleration

### Install Package

```bash
npm install @rUv/neural-trader-backend
```

Or with yarn:

```bash
yarn add @rUv/neural-trader-backend
```

### Verify Installation

```javascript
const { getSystemInfo, getVersion } = require('@rUv/neural-trader-backend');

console.log('Version:', getVersion());
console.log('System Info:', getSystemInfo());
```

Expected output:
```
Version: 2.1.1
System Info: {
  version: '2.1.1',
  rustVersion: '1.75.0',
  buildTimestamp: '2025-01-15T10:00:00Z',
  features: [ 'trading', 'neural', 'sports-betting', 'syndicates', 'e2b-swarm' ],
  totalTools: 70
}
```

---

## First Steps

### 1. Initialize the System

Always initialize before using the API:

```javascript
const { initNeuralTrader } = require('@rUv/neural-trader-backend');

async function setup() {
  const result = await initNeuralTrader();
  console.log(result);  // "Neural Trader initialized successfully"
}

setup();
```

### 2. Check Available Strategies

```javascript
const { listStrategies } = require('@rUv/neural-trader-backend');

async function checkStrategies() {
  const strategies = await listStrategies();

  console.log('Available Trading Strategies:\n');
  strategies.forEach(s => {
    console.log(`- ${s.name}`);
    console.log(`  ${s.description}`);
    console.log(`  GPU Capable: ${s.gpuCapable}\n`);
  });
}

checkStrategies();
```

### 3. Your First Analysis

```javascript
const { quickAnalysis } = require('@rUv/neural-trader-backend');

async function analyzeMarket() {
  const analysis = await quickAnalysis('AAPL', true);

  console.log('Market Analysis for AAPL:');
  console.log(`- Trend: ${analysis.trend}`);
  console.log(`- Volatility: ${analysis.volatility}`);
  console.log(`- Volume Trend: ${analysis.volumeTrend}`);
  console.log(`- Recommendation: ${analysis.recommendation}`);
}

analyzeMarket();
```

---

## Basic Concepts

### Trading Strategies

The API supports multiple trading strategies:

- **Momentum**: Trend-following strategy
- **Mean Reversion**: Buy low, sell high
- **Pairs Trading**: Statistical arbitrage
- **Neural**: AI-powered predictions
- **Arbitrage**: Risk-free profit opportunities

### GPU Acceleration

Many functions support GPU acceleration for faster processing:

```javascript
// Without GPU
const result = await quickAnalysis('AAPL', false);

// With GPU (10-100x faster)
const result = await quickAnalysis('AAPL', true);
```

### Async/Await

All trading functions are asynchronous:

```javascript
// ✓ Correct
async function trade() {
  const result = await executeTrade(/*...*/);
}

// ✗ Wrong - missing await
function trade() {
  const result = executeTrade(/*...*/);  // Returns Promise
}
```

### Error Handling

Always use try-catch for trading operations:

```javascript
async function safeTrade() {
  try {
    const result = await executeTrade('momentum', 'AAPL', 'buy', 100);
    console.log('Trade successful:', result);
  } catch (error) {
    console.error('Trade failed:', error.message);

    // Handle specific errors
    if (error.message.includes('Insufficient funds')) {
      // Add more capital
    } else if (error.message.includes('Rate limit')) {
      // Wait and retry
    }
  }
}
```

---

## Your First Trade

### Step-by-Step Trading Workflow

```javascript
const {
  initNeuralTrader,
  quickAnalysis,
  simulateTrade,
  executeTrade,
  getPortfolioStatus
} = require('@rUv/neural-trader-backend');

async function firstTrade() {
  console.log('=== Your First Trade ===\n');

  // Step 1: Initialize
  console.log('1. Initializing system...');
  await initNeuralTrader();

  // Step 2: Analyze the market
  console.log('\n2. Analyzing AAPL...');
  const analysis = await quickAnalysis('AAPL', true);

  console.log(`   Trend: ${analysis.trend}`);
  console.log(`   Recommendation: ${analysis.recommendation}`);

  if (analysis.recommendation !== 'BUY') {
    console.log('\n   Not a buy signal - stopping here');
    return;
  }

  // Step 3: Simulate first
  console.log('\n3. Simulating trade...');
  const simulation = await simulateTrade(
    'momentum',
    'AAPL',
    'buy',
    true  // Use GPU
  );

  console.log(`   Expected Return: ${simulation.expectedReturn}%`);
  console.log(`   Risk Score: ${simulation.riskScore}`);

  // Step 4: Check if simulation looks good
  if (simulation.expectedReturn < 5 || simulation.riskScore > 0.5) {
    console.log('\n   Simulation not favorable - stopping');
    return;
  }

  // Step 5: Execute the trade
  console.log('\n4. Executing trade...');
  const trade = await executeTrade(
    'momentum',  // Strategy
    'AAPL',      // Symbol
    'buy',       // Action
    100          // Quantity
  );

  console.log('\n   ✓ Trade executed successfully!');
  console.log(`   Order ID: ${trade.orderId}`);
  console.log(`   Fill Price: $${trade.fillPrice}`);
  console.log(`   Status: ${trade.status}`);

  // Step 6: Check portfolio
  console.log('\n5. Updated Portfolio:');
  const portfolio = await getPortfolioStatus(true);

  console.log(`   Total Value: $${portfolio.totalValue.toLocaleString()}`);
  console.log(`   Positions: ${portfolio.positions}`);
  console.log(`   Daily P&L: $${portfolio.dailyPnl.toLocaleString()}`);

  console.log('\n=== Trade Complete ===');
}

firstTrade();
```

---

## Common Patterns

### 1. Market Scanning

```javascript
async function scanMarkets() {
  const symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'];

  for (const symbol of symbols) {
    const analysis = await quickAnalysis(symbol, true);

    if (analysis.recommendation === 'BUY') {
      console.log(`✓ ${symbol}: ${analysis.trend} - BUY SIGNAL`);
    }
  }
}
```

### 2. Strategy Comparison

```javascript
async function compareStrategies() {
  const strategies = ['momentum', 'mean_reversion', 'neural'];
  const symbol = 'AAPL';

  for (const strategy of strategies) {
    const sim = await simulateTrade(strategy, symbol, 'buy', true);
    console.log(`${strategy}: ${sim.expectedReturn}% return, ${sim.riskScore} risk`);
  }
}
```

### 3. Batch Analysis

```javascript
async function batchAnalysis(symbols) {
  const results = await Promise.all(
    symbols.map(s => quickAnalysis(s, true))
  );

  results.forEach((analysis, i) => {
    console.log(`${symbols[i]}: ${analysis.trend} - ${analysis.recommendation}`);
  });
}

batchAnalysis(['AAPL', 'GOOGL', 'MSFT']);
```

### 4. Portfolio Monitoring

```javascript
async function monitorPortfolio(intervalMinutes = 5) {
  setInterval(async () => {
    const portfolio = await getPortfolioStatus(true);

    console.log(`[${new Date().toISOString()}]`);
    console.log(`Total Value: $${portfolio.totalValue.toLocaleString()}`);
    console.log(`Daily P&L: $${portfolio.dailyPnl.toLocaleString()}`);

    // Alert on large loss
    if (portfolio.dailyPnl < -5000) {
      console.log('⚠ ALERT: Daily loss exceeds $5,000!');
    }
  }, intervalMinutes * 60 * 1000);
}

monitorPortfolio(5);
```

---

## Troubleshooting

### Common Issues

#### 1. "Cannot find module"

**Problem:** Package not installed correctly

**Solution:**
```bash
npm install @rUv/neural-trader-backend --force
```

#### 2. "Invalid API key"

**Problem:** Missing or incorrect authentication

**Solution:**
```javascript
const { initAuth, createApiKey } = require('@rUv/neural-trader-backend');

initAuth('your-secret-key');
const apiKey = createApiKey('username', 'user', 100, 365);
```

#### 3. "Rate limit exceeded"

**Problem:** Too many requests

**Solution:**
```javascript
const { initRateLimiter, checkRateLimit } = require('@rUv/neural-trader-backend');

initRateLimiter({ maxRequestsPerMinute: 100 });

if (checkRateLimit('user-id')) {
  // Proceed
} else {
  // Wait
  await new Promise(r => setTimeout(r, 1000));
}
```

#### 4. "GPU not available"

**Problem:** CUDA not installed or GPU not detected

**Solution:**
```javascript
// Fall back to CPU
const result = await quickAnalysis('AAPL', false);  // CPU only
```

#### 5. "Insufficient funds"

**Problem:** Not enough capital for trade

**Solution:**
```javascript
const portfolio = await getPortfolioStatus();
console.log(`Available cash: $${portfolio.cash}`);

// Adjust trade size
const quantity = Math.floor(portfolio.cash / 175);  // Estimate price
```

### Debug Mode

Enable detailed logging:

```javascript
const { initNeuralTrader } = require('@rUv/neural-trader-backend');

// Initialize with debug config
await initNeuralTrader(JSON.stringify({
  log_level: 'debug',
  console_output: true
}));
```

### Getting Help

1. Check the [API Reference](/docs/api-reference/complete-api-reference.md)
2. Review [Examples](/docs/examples/)
3. Search [GitHub Issues](https://github.com/ruvnet/neural-trader/issues)
4. Ask in [Discussions](https://github.com/ruvnet/neural-trader/discussions)

---

## Next Steps

### Beginner Track

1. ✓ Install and setup
2. ✓ First trade
3. → [Trading Examples](/docs/examples/trading-examples.md)
4. → [Portfolio Management](/docs/examples/trading-examples.md#multi-asset-portfolio-management)

### Intermediate Track

1. → [Neural Networks](/docs/examples/neural-examples.md)
2. → [Sports Betting](/docs/examples/syndicate-examples.md)
3. → [Syndicates](/docs/examples/syndicate-examples.md#creating-a-basic-syndicate)

### Advanced Track

1. → [E2B Swarms](/docs/examples/swarm-examples.md)
2. → [Best Practices](/docs/guides/best-practices.md)
3. → [Security](/docs/guides/security.md)

---

## Learning Resources

### Documentation

- [Complete API Reference](/docs/api-reference/complete-api-reference.md)
- [Trading Examples](/docs/examples/trading-examples.md)
- [Neural Network Examples](/docs/examples/neural-examples.md)
- [Syndicate Examples](/docs/examples/syndicate-examples.md)
- [Swarm Examples](/docs/examples/swarm-examples.md)

### Video Tutorials

*(Coming soon)*

### Community

- GitHub Discussions
- Discord Server
- Twitter: @neural_trader

---

## Quick Reference

### Essential Functions

```javascript
// System
initNeuralTrader()
getSystemInfo()
healthCheck()

// Trading
quickAnalysis(symbol, useGpu)
simulateTrade(strategy, symbol, action, useGpu)
executeTrade(strategy, symbol, action, quantity)
getPortfolioStatus(includeAnalytics)

// Strategies
listStrategies()
getStrategyInfo(strategy)
runBacktest(strategy, symbol, startDate, endDate, useGpu)

// Neural
neuralForecast(symbol, horizon, useGpu, confidenceLevel)
neuralTrain(dataPath, modelType, epochs, useGpu)
neuralEvaluate(modelId, testData, useGpu)

// Risk
riskAnalysis(portfolio, useGpu)
portfolioRebalance(targetAllocations, currentPortfolio)
correlationAnalysis(symbols, useGpu)
```

---

**Happy Trading!** 🚀

For questions or support, see our [GitHub repository](https://github.com/ruvnet/neural-trader).
