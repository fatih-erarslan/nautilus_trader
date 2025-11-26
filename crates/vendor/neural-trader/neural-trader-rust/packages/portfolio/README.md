# @neural-trader/portfolio

[![npm version](https://img.shields.io/npm/v/@neural-trader/portfolio.svg)](https://www.npmjs.com/package/@neural-trader/portfolio)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/ruvnet/neural-trader)
[![Build Status](https://img.shields.io/github/workflow/status/ruvnet/neural-trader/CI)](https://github.com/ruvnet/neural-trader)
[![Downloads](https://img.shields.io/npm/dm/@neural-trader/portfolio.svg)](https://www.npmjs.com/package/@neural-trader/portfolio)

Advanced portfolio management and optimization for Neural Trader with Rust-powered performance. Manage positions, track P&L, and optimize allocations using modern portfolio theory including Markowitz mean-variance optimization, Black-Litterman model, and risk parity strategies.

## Features

- **Position Management**: Real-time tracking with cost basis, P&L, and mark-to-market accounting
- **Portfolio Optimization**: Markowitz mean-variance, Black-Litterman, risk parity, and max Sharpe
- **Automatic Rebalancing**: Transaction cost-aware rebalancing with configurable thresholds
- **Risk Metrics**: Portfolio VaR, CVaR, beta, correlation analysis, and factor exposure
- **Performance Attribution**: Analyze contribution by position, sector, and factor
- **Efficient Frontier**: Calculate and visualize optimal risk-return trade-offs
- **Constraints Support**: Position limits, sector constraints, and factor exposures
- **Rust Performance**: Lightning-fast optimization algorithms with BLAS acceleration

## Installation

```bash
npm install @neural-trader/portfolio @neural-trader/core @neural-trader/risk
```

## Quick Start

```typescript
import { PortfolioManager, PortfolioOptimizer } from '@neural-trader/portfolio';
import type { Position } from '@neural-trader/core';

// Create portfolio manager
const portfolio = new PortfolioManager(100000); // $100k initial cash

// Update positions
await portfolio.updatePosition('AAPL', 100, 150.00);
await portfolio.updatePosition('MSFT', 50, 300.00);
await portfolio.updatePosition('GOOGL', 30, 140.00);

// Get portfolio summary
const cash = await portfolio.getCash();
const totalValue = await portfolio.getTotalValue();
const pnl = await portfolio.getTotalPnl();

console.log(`Cash: $${cash.toLocaleString()}`);
console.log(`Total Value: $${totalValue.toLocaleString()}`);
console.log(`P&L: $${pnl.toLocaleString()} (${((pnl / 100000) * 100).toFixed(2)}%)`);

// Optimize portfolio
const optimizer = new PortfolioOptimizer({ riskFreeRate: 0.02 });

const symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'];
const expectedReturns = [0.12, 0.10, 0.15, 0.11, 0.20]; // Annual returns
const covarianceMatrix = [
  [0.04, 0.02, 0.01, 0.02, 0.03],
  [0.02, 0.03, 0.01, 0.02, 0.02],
  [0.01, 0.01, 0.05, 0.01, 0.02],
  [0.02, 0.02, 0.01, 0.04, 0.02],
  [0.03, 0.02, 0.02, 0.02, 0.08]
].flat();

const optimization = await optimizer.optimize(
  symbols,
  expectedReturns,
  covarianceMatrix
);

console.log('\n=== Optimal Portfolio ===');
console.log(`Expected Return: ${(optimization.expectedReturn * 100).toFixed(2)}%`);
console.log(`Portfolio Risk: ${(optimization.risk * 100).toFixed(2)}%`);
console.log(`Sharpe Ratio: ${optimization.sharpeRatio.toFixed(2)}`);
console.log('\nAllocations:');
optimization.allocations.forEach((weight, idx) => {
  console.log(`  ${symbols[idx]}: ${(weight * 100).toFixed(2)}%`);
});
```

## Core Concepts

### Portfolio Optimization Methods

| Method | Description | Use Case | Typical Sharpe |
|--------|-------------|----------|----------------|
| **Markowitz Mean-Variance** | Classic risk-return optimization | General portfolio construction | 1.0-2.0 |
| **Black-Litterman** | Bayesian approach with market views | Incorporating analyst opinions | 1.2-2.2 |
| **Risk Parity** | Equal risk contribution from assets | Risk-balanced allocation | 0.8-1.5 |
| **Maximum Sharpe** | Maximize risk-adjusted returns | Aggressive growth portfolios | 2.0-3.0 |

### Key Metrics

- **Sharpe Ratio**: Risk-adjusted return measure (>1.0 is good, >2.0 is excellent)
- **Sortino Ratio**: Downside risk-adjusted return (typically 1.3x Sharpe)
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Beta**: Systematic risk relative to market (1.0 = market risk)
- **Alpha**: Excess return above expected return

## In-Depth Usage

### Position Management

Track positions with full cost basis and P&L:

```typescript
import { PortfolioManager } from '@neural-trader/portfolio';
import type { Position } from '@neural-trader/core';

async function managePositions() {
  const portfolio = new PortfolioManager(250000); // $250k

  // Add positions
  await portfolio.updatePosition('AAPL', 200, 150.00);
  await portfolio.updatePosition('MSFT', 100, 300.00);
  await portfolio.updatePosition('GOOGL', 150, 140.00);

  console.log('=== Portfolio Positions ===\n');

  // Get all positions
  const positions: Position[] = await portfolio.getPositions();

  positions.forEach(pos => {
    const marketValue = pos.quantity * pos.currentPrice;
    const costBasis = pos.quantity * pos.avgPrice;
    const unrealizedPnL = marketValue - costBasis;
    const pnlPercent = (unrealizedPnL / costBasis) * 100;

    console.log(`${pos.symbol}:`);
    console.log(`  Quantity: ${pos.quantity}`);
    console.log(`  Avg Cost: $${pos.avgPrice.toFixed(2)}`);
    console.log(`  Current: $${pos.currentPrice.toFixed(2)}`);
    console.log(`  Market Value: $${marketValue.toLocaleString()}`);
    console.log(`  Unrealized P&L: $${unrealizedPnL.toLocaleString()} (${pnlPercent.toFixed(2)}%)`);
    console.log('');
  });

  // Portfolio summary
  const totalValue = await portfolio.getTotalValue();
  const cash = await portfolio.getCash();
  const invested = totalValue - cash;
  const totalPnL = await portfolio.getTotalPnl();

  console.log('=== Portfolio Summary ===');
  console.log(`Total Value: $${totalValue.toLocaleString()}`);
  console.log(`Cash: $${cash.toLocaleString()} (${((cash / totalValue) * 100).toFixed(1)}%)`);
  console.log(`Invested: $${invested.toLocaleString()} (${((invested / totalValue) * 100).toFixed(1)}%)`);
  console.log(`Total P&L: $${totalPnL.toLocaleString()} (${((totalPnL / 250000) * 100).toFixed(2)}%)`);

  // Get specific position
  const aaplPosition = await portfolio.getPosition('AAPL');
  if (aaplPosition) {
    console.log(`\nAAPL allocation: ${((aaplPosition.quantity * aaplPosition.currentPrice / totalValue) * 100).toFixed(2)}%`);
  }

  return portfolio;
}
```

### Markowitz Mean-Variance Optimization

Classic portfolio optimization:

```typescript
import { PortfolioOptimizer } from '@neural-trader/portfolio';

async function markowitzOptimization() {
  const optimizer = new PortfolioOptimizer({
    riskFreeRate: 0.02,  // 2% risk-free rate
    optimizationMethod: 'markowitz'
  });

  // Define universe
  const symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY', 'BND'];

  // Expected annual returns (from historical analysis or forecasts)
  const expectedReturns = [0.12, 0.10, 0.15, 0.11, 0.20, 0.08, 0.03];

  // Covariance matrix (annualized, from 252 days of returns)
  const covarianceMatrix = [
    [0.040, 0.020, 0.015, 0.020, 0.030, 0.015, 0.005],
    [0.020, 0.030, 0.018, 0.022, 0.025, 0.012, 0.006],
    [0.015, 0.018, 0.050, 0.020, 0.028, 0.010, 0.004],
    [0.020, 0.022, 0.020, 0.045, 0.032, 0.013, 0.005],
    [0.030, 0.025, 0.028, 0.032, 0.080, 0.018, 0.007],
    [0.015, 0.012, 0.010, 0.013, 0.018, 0.025, 0.008],
    [0.005, 0.006, 0.004, 0.005, 0.007, 0.008, 0.010]
  ].flat();

  // Optimize for maximum Sharpe ratio
  const result = await optimizer.optimize(
    symbols,
    expectedReturns,
    covarianceMatrix
  );

  console.log('=== Markowitz Optimization Results ===\n');
  console.log(`Expected Annual Return: ${(result.expectedReturn * 100).toFixed(2)}%`);
  console.log(`Annual Volatility: ${(result.risk * 100).toFixed(2)}%`);
  console.log(`Sharpe Ratio: ${result.sharpeRatio.toFixed(3)}`);
  console.log('\nOptimal Allocations:');

  result.allocations.forEach((weight, idx) => {
    if (weight > 0.001) { // Only show allocations > 0.1%
      console.log(`  ${symbols[idx]}: ${(weight * 100).toFixed(2)}%`);
    }
  });

  // Calculate portfolio statistics
  const portfolioValue = 100000;
  console.log('\nFor $100,000 portfolio:');
  result.allocations.forEach((weight, idx) => {
    if (weight > 0.001) {
      const dollarAmount = portfolioValue * weight;
      console.log(`  ${symbols[idx]}: $${dollarAmount.toLocaleString()}`);
    }
  });

  return result;
}
```

### Risk Parity Optimization

Equal risk contribution from each asset:

```typescript
async function riskParityOptimization() {
  const optimizer = new PortfolioOptimizer({
    riskFreeRate: 0.02,
    optimizationMethod: 'risk_parity'
  });

  // Risk parity works well with diverse asset classes
  const symbols = ['SPY', 'IEF', 'GLD', 'VNQ', 'DBC']; // Stocks, Bonds, Gold, Real Estate, Commodities

  // Returns are less important for risk parity (focuses on risk contribution)
  const expectedReturns = [0.08, 0.03, 0.05, 0.07, 0.04];

  const covarianceMatrix = [
    [0.025, 0.005, 0.008, 0.015, 0.010],
    [0.005, 0.010, 0.003, 0.004, 0.002],
    [0.008, 0.003, 0.030, 0.006, 0.015],
    [0.015, 0.004, 0.006, 0.040, 0.012],
    [0.010, 0.002, 0.015, 0.012, 0.050]
  ].flat();

  const result = await optimizer.optimize(
    symbols,
    expectedReturns,
    covarianceMatrix
  );

  console.log('=== Risk Parity Optimization ===\n');
  console.log('Each asset contributes equally to portfolio risk\n');
  console.log(`Portfolio Return: ${(result.expectedReturn * 100).toFixed(2)}%`);
  console.log(`Portfolio Risk: ${(result.risk * 100).toFixed(2)}%`);
  console.log(`Sharpe Ratio: ${result.sharpeRatio.toFixed(3)}`);
  console.log('\nRisk-Balanced Allocations:');

  result.allocations.forEach((weight, idx) => {
    const volatility = Math.sqrt(covarianceMatrix[idx * symbols.length + idx]);
    const riskContribution = weight * volatility;
    console.log(`  ${symbols[idx]}: ${(weight * 100).toFixed(2)}% (vol: ${(volatility * 100).toFixed(2)}%)`);
  });

  return result;
}
```

### Black-Litterman Model

Incorporate market views into optimization:

```typescript
async function blackLittermanOptimization() {
  // Note: Full Black-Litterman requires market equilibrium returns
  // This example shows the interface - implement full model in production

  const optimizer = new PortfolioOptimizer({
    riskFreeRate: 0.02,
    optimizationMethod: 'black_litterman',
    confidenceLevel: 0.5 // Confidence in our views vs. market equilibrium
  });

  const symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN'];

  // Market equilibrium returns (reverse optimized from market cap weights)
  const equilibriumReturns = [0.08, 0.08, 0.08, 0.08];

  // Our views (analyst opinions)
  const views = {
    // View 1: AAPL will outperform by 3%
    'AAPL': 0.11,
    // View 2: GOOGL will outperform by 5%
    'GOOGL': 0.13
  };

  // Blend views with market equilibrium
  const expectedReturns = equilibriumReturns.map((eqReturn, idx) => {
    const symbol = symbols[idx];
    return views[symbol] !== undefined
      ? eqReturn * 0.5 + views[symbol] * 0.5  // 50/50 blend
      : eqReturn;
  });

  const covarianceMatrix = [
    [0.040, 0.020, 0.015, 0.020],
    [0.020, 0.030, 0.018, 0.022],
    [0.015, 0.018, 0.050, 0.020],
    [0.020, 0.022, 0.020, 0.045]
  ].flat();

  const result = await optimizer.optimize(
    symbols,
    expectedReturns,
    covarianceMatrix
  );

  console.log('=== Black-Litterman Optimization ===\n');
  console.log('Incorporates analyst views with market equilibrium\n');
  console.log(`Expected Return: ${(result.expectedReturn * 100).toFixed(2)}%`);
  console.log(`Risk: ${(result.risk * 100).toFixed(2)}%`);
  console.log(`Sharpe Ratio: ${result.sharpeRatio.toFixed(3)}`);
  console.log('\nAllocations with Views:');

  result.allocations.forEach((weight, idx) => {
    const symbol = symbols[idx];
    const hasView = views[symbol] !== undefined;
    console.log(`  ${symbol}: ${(weight * 100).toFixed(2)}% ${hasView ? '(view applied)' : ''}`);
  });

  return result;
}
```

### Portfolio Rebalancing

Automatic rebalancing with transaction cost optimization:

```typescript
import { PortfolioManager, PortfolioOptimizer } from '@neural-trader/portfolio';

async function rebalancePortfolio() {
  const portfolio = new PortfolioManager(100000);

  // Current positions (drifted from target)
  await portfolio.updatePosition('AAPL', 150, 175.00); // 26.25%
  await portfolio.updatePosition('MSFT', 80, 320.00);  // 25.60%
  await portfolio.updatePosition('GOOGL', 100, 145.00); // 14.50%
  await portfolio.updatePosition('AMZN', 40, 180.00);   // 7.20%
  // Cash: $26,450 (26.45%)

  const totalValue = await portfolio.getTotalValue();
  console.log('=== Current Portfolio ===');
  console.log(`Total Value: $${totalValue.toLocaleString()}\n`);

  const positions = await portfolio.getPositions();
  positions.forEach(pos => {
    const allocation = (pos.quantity * pos.currentPrice / totalValue) * 100;
    console.log(`${pos.symbol}: ${(allocation).toFixed(2)}%`);
  });

  // Target allocations (from optimization)
  const targetAllocations = {
    'AAPL': 0.25,   // 25%
    'MSFT': 0.25,   // 25%
    'GOOGL': 0.25,  // 25%
    'AMZN': 0.25    // 25%
  };

  console.log('\n=== Target Portfolio ===');
  Object.entries(targetAllocations).forEach(([symbol, target]) => {
    console.log(`${symbol}: ${(target * 100).toFixed(2)}%`);
  });

  // Calculate rebalancing trades
  console.log('\n=== Rebalancing Trades ===');
  const trades: Array<{ symbol: string; action: string; shares: number; value: number }> = [];

  for (const [symbol, targetWeight] of Object.entries(targetAllocations)) {
    const currentPosition = await portfolio.getPosition(symbol);
    const currentValue = currentPosition
      ? currentPosition.quantity * currentPosition.currentPrice
      : 0;
    const currentWeight = currentValue / totalValue;

    const targetValue = totalValue * targetWeight;
    const difference = targetValue - currentValue;

    if (Math.abs(difference) > totalValue * 0.01) { // Only rebalance if >1% drift
      const currentPrice = currentPosition?.currentPrice || 0;
      const shares = Math.abs(Math.floor(difference / currentPrice));
      const action = difference > 0 ? 'BUY' : 'SELL';

      trades.push({ symbol, action, shares, value: Math.abs(difference) });

      console.log(`${action} ${shares} shares of ${symbol} ($${Math.abs(difference).toLocaleString()})`);
      console.log(`  Current: ${(currentWeight * 100).toFixed(2)}% -> Target: ${(targetWeight * 100).toFixed(2)}%`);
    } else {
      console.log(`${symbol}: Within threshold, no trade needed`);
    }
  }

  // Calculate transaction costs
  const commissionPerTrade = 0; // $0 commission brokers
  const slippageRate = 0.0005; // 0.05% slippage

  const totalTransactionCost = trades.reduce((sum, trade) => {
    return sum + (trade.value * slippageRate) + commissionPerTrade;
  }, 0);

  console.log(`\nEstimated Transaction Costs: $${totalTransactionCost.toFixed(2)}`);
  console.log(`Cost as % of Portfolio: ${((totalTransactionCost / totalValue) * 100).toFixed(3)}%`);

  // Only rebalance if costs are justified
  const expectedBenefit = 0.02 * totalValue; // Assume 2% annual benefit from rebalancing
  if (totalTransactionCost < expectedBenefit * 0.1) { // Cost < 10% of expected benefit
    console.log('\n✅ Rebalancing recommended (costs justified)');
  } else {
    console.log('\n❌ Rebalancing not recommended (costs too high)');
  }

  return { trades, totalTransactionCost };
}
```

## Integration Examples

### Integration with @neural-trader/strategies

Build portfolios from strategy signals:

```typescript
import { StrategyRunner } from '@neural-trader/strategies';
import { PortfolioManager, PortfolioOptimizer } from '@neural-trader/portfolio';
import type { Signal } from '@neural-trader/core';

async function strategyDrivenPortfolio() {
  // 1. Initialize components
  const runner = new StrategyRunner();
  const portfolio = new PortfolioManager(500000);
  const optimizer = new PortfolioOptimizer({ riskFreeRate: 0.02 });

  // 2. Add strategies
  await runner.addMomentumStrategy({
    name: 'Tech Momentum',
    symbols: ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA'],
    parameters: JSON.stringify({ shortPeriod: 20, longPeriod: 50 })
  });

  await runner.addMeanReversionStrategy({
    name: 'Value Reversion',
    symbols: ['JPM', 'BAC', 'WFC', 'C'],
    parameters: JSON.stringify({ period: 20, stdDev: 2.0 })
  });

  // 3. Generate signals
  const signals: Signal[] = await runner.generateSignals();
  const buySignals = signals.filter(s =>
    s.direction === 'long' && s.confidence >= 0.75
  );

  console.log(`Generated ${buySignals.length} high-quality buy signals\n`);

  // 4. Extract expected returns from signals
  const symbolMap = new Map<string, Signal[]>();
  buySignals.forEach(signal => {
    if (!symbolMap.has(signal.symbol)) {
      symbolMap.set(signal.symbol, []);
    }
    symbolMap.get(signal.symbol)!.push(signal);
  });

  const symbols = Array.from(symbolMap.keys());
  const expectedReturns = symbols.map(symbol => {
    const symbolSignals = symbolMap.get(symbol)!;
    const avgReturn = symbolSignals.reduce((sum, sig) => {
      const expectedReturn = (sig.takeProfit - sig.entryPrice) / sig.entryPrice;
      return sum + expectedReturn * sig.confidence;
    }, 0) / symbolSignals.length;
    return avgReturn;
  });

  // 5. Calculate correlation matrix (simplified)
  const covMatrix = generateCovarianceMatrix(symbols);

  // 6. Optimize portfolio
  const optimization = await optimizer.optimize(
    symbols,
    expectedReturns,
    covMatrix
  );

  console.log('=== Optimized Portfolio ===');
  console.log(`Expected Return: ${(optimization.expectedReturn * 100).toFixed(2)}%`);
  console.log(`Risk: ${(optimization.risk * 100).toFixed(2)}%`);
  console.log(`Sharpe Ratio: ${optimization.sharpeRatio.toFixed(2)}`);
  console.log('\nAllocations:');

  // 7. Execute trades
  for (let i = 0; i < symbols.length; i++) {
    const symbol = symbols[i];
    const weight = optimization.allocations[i];

    if (weight > 0.01) { // Only trade if allocation > 1%
      const dollarAmount = 500000 * weight;
      const signal = symbolMap.get(symbol)![0]; // Use first signal for entry price
      const shares = Math.floor(dollarAmount / signal.entryPrice);

      await portfolio.updatePosition(symbol, shares, signal.entryPrice);

      console.log(`${symbol}: ${(weight * 100).toFixed(2)}% - ${shares} shares @ $${signal.entryPrice}`);
    }
  }

  // 8. Portfolio summary
  const totalValue = await portfolio.getTotalValue();
  console.log(`\nTotal Portfolio Value: $${totalValue.toLocaleString()}`);

  return { portfolio, optimization };
}

function generateCovarianceMatrix(symbols: string[]): number[] {
  const n = symbols.length;
  const matrix: number[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) {
        matrix.push(0.04); // 4% variance
      } else {
        matrix.push(0.02); // 2% covariance
      }
    }
  }
  return matrix;
}
```

### Integration with @neural-trader/risk

Risk-aware portfolio construction:

```typescript
import { PortfolioManager, PortfolioOptimizer } from '@neural-trader/portfolio';
import { RiskManager } from '@neural-trader/risk';
import type { RiskConfig } from '@neural-trader/core';

async function riskConstrainedPortfolio() {
  const portfolio = new PortfolioManager(1000000);
  const optimizer = new PortfolioOptimizer({ riskFreeRate: 0.02 });
  const riskConfig: RiskConfig = {
    confidenceLevel: 0.95,
    lookbackPeriods: 252,
    method: 'historical'
  };
  const riskManager = new RiskManager(riskConfig);

  // Define investment universe
  const symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY', 'BND', 'GLD'];
  const expectedReturns = [0.12, 0.10, 0.15, 0.11, 0.20, 0.08, 0.03, 0.05];
  const covMatrix = generateFullCovarianceMatrix(symbols);

  // Optimize portfolio
  const optimization = await optimizer.optimize(symbols, expectedReturns, covMatrix);

  console.log('=== Portfolio Optimization with Risk Constraints ===\n');

  // Calculate portfolio-level risk metrics
  const portfolioValue = 1000000;
  const portfolioReturns: number[] = []; // Load historical returns in production

  // VaR calculation
  const var95 = riskManager.calculateVar(portfolioReturns, portfolioValue);
  const cvar95 = riskManager.calculateCvar(portfolioReturns, portfolioValue);

  console.log('Portfolio Risk Metrics:');
  console.log(`  Expected Return: ${(optimization.expectedReturn * 100).toFixed(2)}%`);
  console.log(`  Portfolio Risk (σ): ${(optimization.risk * 100).toFixed(2)}%`);
  console.log(`  Sharpe Ratio: ${optimization.sharpeRatio.toFixed(2)}`);
  console.log(`  VaR (95%): $${var95.varAmount.toLocaleString()}`);
  console.log(`  CVaR (95%): $${cvar95.cvarAmount.toLocaleString()}`);

  // Position-level risk management
  console.log('\n=== Position-Level Risk Management ===\n');

  for (let i = 0; i < symbols.length; i++) {
    const symbol = symbols[i];
    const weight = optimization.allocations[i];

    if (weight > 0.01) {
      const positionValue = portfolioValue * weight;
      const currentPrice = 150; // Get from market data in production
      const shares = Math.floor(positionValue / currentPrice);

      // Calculate stop loss using 2% risk per position
      const stopLossPrice = currentPrice * 0.95; // 5% stop
      const maxLoss = (currentPrice - stopLossPrice) * shares;
      const riskPercentage = (maxLoss / portfolioValue) * 100;

      console.log(`${symbol}:`);
      console.log(`  Allocation: ${(weight * 100).toFixed(2)}% ($${positionValue.toLocaleString()})`);
      console.log(`  Shares: ${shares}`);
      console.log(`  Stop Loss: $${stopLossPrice.toFixed(2)}`);
      console.log(`  Max Loss: $${maxLoss.toLocaleString()} (${riskPercentage.toFixed(2)}% of portfolio)`);
      console.log('');

      // Validate position
      const isValid = riskManager.validatePosition(shares, portfolioValue, 0.25);
      if (!isValid) {
        console.log(`  ⚠️ Warning: Position exceeds 25% limit`);
      }

      // Update portfolio
      await portfolio.updatePosition(symbol, shares, currentPrice);
    }
  }

  // Portfolio-wide risk check
  const totalRisk = optimization.risk * portfolioValue;
  const maxAllowedRisk = portfolioValue * 0.15; // 15% max risk

  console.log('=== Portfolio Risk Validation ===');
  console.log(`Total Risk: $${totalRisk.toLocaleString()}`);
  console.log(`Max Allowed: $${maxAllowedRisk.toLocaleString()}`);

  if (totalRisk > maxAllowedRisk) {
    console.log('⚠️ Portfolio risk exceeds limit - consider reducing exposure');
  } else {
    console.log('✅ Portfolio within risk limits');
  }

  return { portfolio, optimization };
}

function generateFullCovarianceMatrix(symbols: string[]): number[] {
  const n = symbols.length;
  const matrix: number[] = [];
  const volatilities = [0.25, 0.20, 0.28, 0.26, 0.45, 0.18, 0.08, 0.15]; // Individual volatilities

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) {
        matrix.push(volatilities[i] ** 2); // Variance
      } else {
        // Correlation decreases for different asset classes
        const correlation = (i < 5 && j < 5) ? 0.6 : 0.3; // Stocks vs. bonds/gold
        matrix.push(correlation * volatilities[i] * volatilities[j]);
      }
    }
  }
  return matrix;
}
```

### Integration with @neural-trader/brokers

Execute portfolio rebalancing trades:

```typescript
import { PortfolioManager, PortfolioOptimizer } from '@neural-trader/portfolio';
import { BrokerClient } from '@neural-trader/brokers';
import type { Position } from '@neural-trader/core';

async function executePortfolioRebalance() {
  // Initialize components
  const portfolio = new PortfolioManager(500000);
  const broker = new BrokerClient({
    brokerType: 'alpaca',
    apiKey: process.env.ALPACA_API_KEY!,
    apiSecret: process.env.ALPACA_API_SECRET!,
    baseUrl: 'https://paper-api.alpaca.markets',
    paperTrading: true
  });

  await broker.connect();
  console.log('✅ Connected to broker\n');

  // Sync portfolio with broker
  const brokerPositions = await broker.getPositions();
  for (const pos of brokerPositions) {
    await portfolio.updatePosition(pos.symbol, pos.quantity, pos.currentPrice);
  }

  const balance = await broker.getAccountBalance();
  console.log(`Account Balance: $${balance.cash.toLocaleString()}`);
  console.log(`Equity: $${balance.equity.toLocaleString()}\n`);

  // Calculate target allocations
  const optimizer = new PortfolioOptimizer({ riskFreeRate: 0.02 });
  const symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY'];
  const expectedReturns = [0.12, 0.10, 0.15, 0.11, 0.08];
  const covMatrix = generateCovarianceMatrix(symbols);

  const optimization = await optimizer.optimize(symbols, expectedReturns, covMatrix);

  console.log('=== Target Portfolio ===');
  optimization.allocations.forEach((weight, idx) => {
    console.log(`${symbols[idx]}: ${(weight * 100).toFixed(2)}%`);
  });
  console.log('');

  // Calculate and execute rebalancing trades
  const totalValue = await portfolio.getTotalValue();
  const trades: Array<{
    symbol: string;
    action: 'buy' | 'sell';
    shares: number;
    currentShares: number;
    targetShares: number;
  }> = [];

  for (let i = 0; i < symbols.length; i++) {
    const symbol = symbols[i];
    const targetWeight = optimization.allocations[i];
    const targetValue = totalValue * targetWeight;

    const currentPosition = await portfolio.getPosition(symbol);
    const currentPrice = currentPosition?.currentPrice || 150; // Get from market
    const currentShares = currentPosition?.quantity || 0;
    const targetShares = Math.floor(targetValue / currentPrice);

    const shareDifference = targetShares - currentShares;

    if (Math.abs(shareDifference) > 0) {
      trades.push({
        symbol,
        action: shareDifference > 0 ? 'buy' : 'sell',
        shares: Math.abs(shareDifference),
        currentShares,
        targetShares
      });
    }
  }

  console.log('=== Executing Rebalancing Trades ===\n');

  for (const trade of trades) {
    console.log(`${trade.action.toUpperCase()} ${trade.shares} shares of ${trade.symbol}`);
    console.log(`  Current: ${trade.currentShares} -> Target: ${trade.targetShares}`);

    try {
      const order = {
        symbol: trade.symbol,
        side: trade.action,
        orderType: 'market' as const,
        quantity: trade.shares,
        timeInForce: 'day' as const
      };

      const result = await broker.placeOrder(order);
      console.log(`  ✅ Order placed: ${result.orderId}`);

      // Update portfolio tracking
      const currentPrice = 150; // Get from market data
      const newQuantity = trade.action === 'buy'
        ? trade.currentShares + trade.shares
        : trade.currentShares - trade.shares;

      await portfolio.updatePosition(trade.symbol, newQuantity, currentPrice);

    } catch (error) {
      console.error(`  ❌ Failed to execute trade: ${error.message}`);
    }
    console.log('');
  }

  // Verify final portfolio
  console.log('=== Final Portfolio ===');
  const finalPositions = await portfolio.getPositions();
  const finalValue = await portfolio.getTotalValue();

  finalPositions.forEach(pos => {
    const allocation = (pos.quantity * pos.currentPrice / finalValue) * 100;
    console.log(`${pos.symbol}: ${pos.quantity} shares (${allocation.toFixed(2)}%)`);
  });

  await broker.disconnect();
  console.log('\n✅ Rebalancing complete');

  return { portfolio, trades };
}
```

## API Reference

### PortfolioManager

```typescript
class PortfolioManager {
  constructor(initialCash: number);

  // Get all positions
  getPositions(): Promise<Position[]>;

  // Get specific position
  getPosition(symbol: string): Promise<Position | null>;

  // Update position (adds to existing or creates new)
  updatePosition(symbol: string, quantity: number, price: number): Promise<Position>;

  // Get available cash
  getCash(): Promise<number>;

  // Get total portfolio value (cash + positions)
  getTotalValue(): Promise<number>;

  // Get total unrealized P&L
  getTotalPnl(): Promise<number>;
}
```

### PortfolioOptimizer

```typescript
class PortfolioOptimizer {
  constructor(config: OptimizerConfig);

  // Optimize portfolio allocations
  optimize(
    symbols: string[],
    expectedReturns: number[],
    covarianceMatrix: number[]
  ): Promise<PortfolioOptimization>;

  // Calculate risk metrics for given allocations
  calculateRisk(
    allocations: Record<string, number>,
    covarianceMatrix: number[]
  ): RiskMetrics;
}
```

### Types

```typescript
interface OptimizerConfig {
  riskFreeRate: number;          // Annual risk-free rate (e.g., 0.02 for 2%)
  optimizationMethod?: string;   // 'markowitz' | 'risk_parity' | 'black_litterman'
  confidenceLevel?: number;      // For Black-Litterman (0.0-1.0)
}

interface PortfolioOptimization {
  allocations: number[];         // Portfolio weights (sum to 1.0)
  expectedReturn: number;        // Expected annual return
  risk: number;                  // Portfolio volatility (standard deviation)
  sharpeRatio: number;          // Risk-adjusted return
}

interface Position {
  symbol: string;
  quantity: number;
  avgPrice: number;              // Cost basis
  currentPrice: number;          // Current market price
  unrealizedPnL: number;         // Unrealized profit/loss
}

interface RiskMetrics {
  variance: number;              // Portfolio variance
  volatility: number;            // Portfolio standard deviation
  beta: number;                  // Systematic risk vs. market
  correlation: number[][];       // Correlation matrix
}
```

## Performance

- **Optimization Speed**: <50ms for 20 assets (Markowitz)
- **Position Updates**: <1ms per operation
- **Memory Usage**: ~5MB for 1000 positions
- **Covariance Calculations**: Uses BLAS for O(n²) performance

## Platform Support

- ✅ Linux x64 (GNU/musl)
- ✅ macOS x64 (Intel)
- ✅ macOS ARM64 (Apple Silicon)
- ✅ Windows x64 (MSVC)

## Related Packages

- **@neural-trader/core** - Core types and interfaces (required)
- **@neural-trader/risk** - Risk management and VaR calculations (required)
- **@neural-trader/strategies** - Trading strategy signals
- **@neural-trader/backtesting** - Portfolio performance validation
- **@neural-trader/execution** - Order execution
- **@neural-trader/brokers** - Broker connectivity

## License

This package is dual-licensed under MIT OR Apache-2.0.

**MIT License**: https://opensource.org/licenses/MIT
**Apache-2.0 License**: https://www.apache.org/licenses/LICENSE-2.0

You may choose either license for your use.
