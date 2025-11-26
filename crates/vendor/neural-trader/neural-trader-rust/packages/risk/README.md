# @neural-trader/risk

[![npm version](https://img.shields.io/npm/v/@neural-trader/risk.svg)](https://www.npmjs.com/package/@neural-trader/risk)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/ruvnet/neural-trader)
[![Build Status](https://img.shields.io/github/workflow/status/ruvnet/neural-trader/CI)](https://github.com/ruvnet/neural-trader)
[![Downloads](https://img.shields.io/npm/dm/@neural-trader/risk.svg)](https://www.npmjs.com/package/@neural-trader/risk)

Comprehensive risk management toolkit for the Neural Trader platform, powered by Rust with native Node.js bindings. Calculate Value at Risk (VaR), Conditional VaR, Kelly Criterion position sizing, drawdown metrics, and Sharpe/Sortino ratios with blazing-fast performance.

## Features

- **Value at Risk (VaR)**: Historical, parametric, and Monte Carlo methods
- **Conditional VaR (CVaR)**: Expected shortfall calculations
- **Kelly Criterion**: Optimal position sizing based on edge and odds
- **Drawdown Analysis**: Maximum drawdown, duration, and recovery metrics
- **Risk-Adjusted Returns**: Sharpe ratio, Sortino ratio, and more
- **Position Sizing**: Calculate optimal position sizes based on risk parameters
- **Leverage Management**: Maximum leverage calculations based on volatility
- **Rust Performance**: Native implementation delivers microsecond-level calculations

## Installation

```bash
npm install @neural-trader/risk @neural-trader/core
```

## Quick Start

```typescript
import { RiskManager, calculateSharpeRatio } from '@neural-trader/risk';
import type { RiskConfig } from '@neural-trader/core';

// Configure risk manager
const config: RiskConfig = {
  confidenceLevel: 0.95,  // 95% confidence
  lookbackPeriods: 252,   // 1 year of daily data
  method: 'historical'    // Historical VaR method
};

const riskManager = new RiskManager(config);

// Calculate VaR
const returns = [0.01, -0.02, 0.015, -0.01, 0.03, /* ... */];
const portfolioValue = 100000;
const var95 = riskManager.calculateVar(returns, portfolioValue);

console.log(`VaR (95%): $${var95.varAmount.toFixed(2)}`);
console.log(`Risk: ${(var95.varPercentage * 100).toFixed(2)}% of portfolio`);

// Calculate CVaR (Expected Shortfall)
const cvar = riskManager.calculateCvar(returns, portfolioValue);
console.log(`CVaR (95%): $${cvar.cvarAmount.toFixed(2)}`);
console.log(`Expected loss in worst ${((1 - config.confidenceLevel) * 100).toFixed(0)}% of cases`);

// Kelly Criterion position sizing
const kelly = riskManager.calculateKelly(0.55, 1.8, 1.0);
console.log(`Kelly Fraction: ${(kelly.kellyFraction * 100).toFixed(2)}%`);
console.log(`Half Kelly (recommended): ${(kelly.halfKelly * 100).toFixed(2)}%`);

// Calculate Sharpe ratio
const sharpe = calculateSharpeRatio(returns, 0.02, 252);
console.log(`Sharpe Ratio: ${sharpe.toFixed(2)}`);
```

## In-Depth Usage

### Value at Risk (VaR) Calculation

```typescript
import { RiskManager } from '@neural-trader/risk';
import type { VaRResult } from '@neural-trader/core';

// Historical VaR (non-parametric)
const historicalConfig = {
  confidenceLevel: 0.95,
  lookbackPeriods: 252,
  method: 'historical'
};

const riskManager = new RiskManager(historicalConfig);

// Portfolio returns (daily)
const dailyReturns = [
  0.012, -0.008, 0.015, -0.023, 0.008,
  0.003, -0.015, 0.020, -0.012, 0.005,
  // ... 252 days of returns
];

const portfolioValue = 500000;
const varResult: VaRResult = riskManager.calculateVar(dailyReturns, portfolioValue);

console.log('Value at Risk Analysis:');
console.log('='.repeat(50));
console.log(`Confidence Level: ${(varResult.confidenceLevel * 100).toFixed(0)}%`);
console.log(`Method: ${varResult.method}`);
console.log(`Portfolio Value: $${varResult.portfolioValue.toLocaleString()}`);
console.log(`VaR Amount: $${varResult.varAmount.toLocaleString()}`);
console.log(`VaR Percentage: ${(varResult.varPercentage * 100).toFixed(2)}%`);
console.log(`\nInterpretation: With ${(varResult.confidenceLevel * 100).toFixed(0)}% confidence,`);
console.log(`portfolio losses will not exceed $${varResult.varAmount.toLocaleString()} in one day.`);
```

### Conditional Value at Risk (CVaR)

```typescript
import type { CVaRResult } from '@neural-trader/core';

// Calculate CVaR (Expected Shortfall)
const cvarResult: CVaRResult = riskManager.calculateCvar(dailyReturns, portfolioValue);

console.log('\nConditional Value at Risk (Expected Shortfall):');
console.log('='.repeat(50));
console.log(`CVaR Amount: $${cvarResult.cvarAmount.toLocaleString()}`);
console.log(`CVaR Percentage: ${(cvarResult.cvarPercentage * 100).toFixed(2)}%`);
console.log(`VaR Amount: $${cvarResult.varAmount.toLocaleString()}`);
console.log(`\nInterpretation: In the worst ${((1 - cvarResult.confidenceLevel) * 100).toFixed(0)}% of scenarios,`);
console.log(`the average loss is expected to be $${cvarResult.cvarAmount.toLocaleString()}.`);

// CVaR is always >= VaR
const ratio = cvarResult.cvarAmount / cvarResult.varAmount;
console.log(`\nCVaR/VaR Ratio: ${ratio.toFixed(2)}`);
console.log(`This indicates the severity of tail risk.`);
```

### Kelly Criterion Position Sizing

```typescript
import type { KellyResult } from '@neural-trader/core';

// Calculate optimal position size using Kelly Criterion
function calculateOptimalPosition(
  winRate: number,
  avgWin: number,
  avgLoss: number,
  portfolioValue: number,
  currentPrice: number
) {
  const kelly: KellyResult = riskManager.calculateKelly(winRate, avgWin, avgLoss);

  console.log('Kelly Criterion Analysis:');
  console.log('='.repeat(50));
  console.log(`Win Rate: ${(kelly.winRate * 100).toFixed(2)}%`);
  console.log(`Average Win: ${kelly.avgWin.toFixed(2)}x`);
  console.log(`Average Loss: ${kelly.avgLoss.toFixed(2)}x`);
  console.log('');
  console.log(`Full Kelly: ${(kelly.kellyFraction * 100).toFixed(2)}% of portfolio`);
  console.log(`Half Kelly: ${(kelly.halfKelly * 100).toFixed(2)}% (recommended)`);
  console.log(`Quarter Kelly: ${(kelly.quarterKelly * 100).toFixed(2)}% (conservative)`);

  // Calculate actual position sizes
  const fullKellyAmount = portfolioValue * kelly.kellyFraction;
  const halfKellyAmount = portfolioValue * kelly.halfKelly;
  const quarterKellyAmount = portfolioValue * kelly.quarterKelly;

  console.log('');
  console.log('Position Sizes:');
  console.log(`Full Kelly: $${fullKellyAmount.toLocaleString()} (${Math.floor(fullKellyAmount / currentPrice)} shares)`);
  console.log(`Half Kelly: $${halfKellyAmount.toLocaleString()} (${Math.floor(halfKellyAmount / currentPrice)} shares)`);
  console.log(`Quarter Kelly: $${quarterKellyAmount.toLocaleString()} (${Math.floor(quarterKellyAmount / currentPrice)} shares)`);

  // Risk warning for Full Kelly
  if (kelly.kellyFraction > 0.25) {
    console.log('\n⚠️  Warning: Full Kelly suggests aggressive sizing.');
    console.log('Consider using Half Kelly or Quarter Kelly to reduce volatility.');
  }

  return kelly;
}

// Example usage
const kelly = calculateOptimalPosition(
  0.58,   // 58% win rate
  1.5,    // Average win is 1.5x bet
  1.0,    // Average loss is 1.0x bet
  100000, // $100k portfolio
  150.00  // Stock price
);
```

### Drawdown Analysis

```typescript
import type { DrawdownMetrics } from '@neural-trader/core';

function analyzeDrawdowns(equityCurve: number[]) {
  const drawdown: DrawdownMetrics = riskManager.calculateDrawdown(equityCurve);

  console.log('Drawdown Analysis:');
  console.log('='.repeat(50));
  console.log(`Maximum Drawdown: ${(drawdown.maxDrawdown * 100).toFixed(2)}%`);
  console.log(`Max Drawdown Duration: ${drawdown.maxDrawdownDuration} periods`);
  console.log(`Current Drawdown: ${(drawdown.currentDrawdown * 100).toFixed(2)}%`);
  console.log(`Recovery Factor: ${drawdown.recoveryFactor.toFixed(2)}`);

  // Calculate additional metrics
  const peak = Math.max(...equityCurve);
  const trough = peak * (1 - drawdown.maxDrawdown);
  const finalEquity = equityCurve[equityCurve.length - 1];

  console.log('');
  console.log('Additional Metrics:');
  console.log(`Peak Equity: $${peak.toLocaleString()}`);
  console.log(`Trough Equity: $${trough.toLocaleString()}`);
  console.log(`Drawdown Amount: $${(peak - trough).toLocaleString()}`);
  console.log(`Current Equity: $${finalEquity.toLocaleString()}`);

  // Status
  if (drawdown.currentDrawdown === 0) {
    console.log('\n✓ Portfolio at all-time high');
  } else {
    const recovered = (1 - drawdown.currentDrawdown / drawdown.maxDrawdown) * 100;
    console.log(`\nRecovered ${recovered.toFixed(1)}% from maximum drawdown`);
  }

  // Risk assessment
  if (drawdown.maxDrawdown > 0.3) {
    console.log('\n⚠️  High Risk: Maximum drawdown exceeds 30%');
  } else if (drawdown.maxDrawdown > 0.2) {
    console.log('\n⚠️  Moderate Risk: Maximum drawdown between 20-30%');
  } else {
    console.log('\n✓ Low Risk: Maximum drawdown under 20%');
  }

  return drawdown;
}

// Example: Analyze backtest equity curve
const equityCurve = [
  100000, 102000, 103500, 101000, 105000,
  108000, 106000, 110000, 108000, 115000
];
analyzeDrawdowns(equityCurve);
```

### Advanced Position Sizing

```typescript
import type { PositionSize } from '@neural-trader/core';

function calculateRiskBasedPosition(
  portfolioValue: number,
  entryPrice: number,
  stopLoss: number,
  riskPerTrade: number = 0.02  // Risk 2% per trade
): PositionSize {
  const stopLossDistance = Math.abs(entryPrice - stopLoss) / entryPrice;

  const positionSize: PositionSize = riskManager.calculatePositionSize(
    portfolioValue,
    entryPrice,
    riskPerTrade,
    stopLossDistance
  );

  console.log('Position Sizing Calculation:');
  console.log('='.repeat(50));
  console.log(`Entry Price: $${entryPrice.toFixed(2)}`);
  console.log(`Stop Loss: $${stopLoss.toFixed(2)}`);
  console.log(`Stop Distance: ${(stopLossDistance * 100).toFixed(2)}%`);
  console.log(`Risk Per Trade: ${(riskPerTrade * 100).toFixed(2)}% of portfolio`);
  console.log('');
  console.log(`Recommended Shares: ${positionSize.shares}`);
  console.log(`Dollar Amount: $${positionSize.dollarAmount.toLocaleString()}`);
  console.log(`Portfolio Allocation: ${(positionSize.percentageOfPortfolio * 100).toFixed(2)}%`);
  console.log(`Maximum Loss if Stopped: $${positionSize.maxLoss.toLocaleString()}`);
  console.log('');
  console.log(`Reasoning: ${positionSize.reasoning}`);

  return positionSize;
}

// Example: Calculate position for a trade
const position = calculateRiskBasedPosition(
  100000,  // $100k portfolio
  150.00,  // Entry at $150
  145.00,  // Stop at $145 (3.33% risk per share)
  0.02     // Risk 2% of portfolio ($2,000)
);

// Validate position size
const isValid = riskManager.validatePosition(
  position.shares,
  100000,
  0.25  // Max 25% portfolio per position
);

if (!isValid) {
  console.log('\n⚠️  Position exceeds maximum allocation limits');
}
```

### Risk-Adjusted Performance Metrics

```typescript
import { calculateSharpeRatio, calculateSortinoRatio } from '@neural-trader/risk';

function analyzeRiskAdjustedReturns(returns: number[]) {
  const riskFreeRate = 0.02;  // 2% annual risk-free rate
  const annualizationFactor = 252;  // Daily returns

  // Sharpe Ratio (penalizes all volatility)
  const sharpe = calculateSharpeRatio(
    returns,
    riskFreeRate,
    annualizationFactor
  );

  // Sortino Ratio (penalizes only downside volatility)
  const sortino = calculateSortinoRatio(
    returns,
    riskFreeRate / 252,  // Daily target return
    annualizationFactor
  );

  console.log('Risk-Adjusted Performance:');
  console.log('='.repeat(50));
  console.log(`Sharpe Ratio: ${sharpe.toFixed(2)}`);
  console.log(`Sortino Ratio: ${sortino.toFixed(2)}`);
  console.log('');

  // Interpretation
  console.log('Sharpe Ratio Interpretation:');
  if (sharpe < 1.0) {
    console.log('  < 1.0: Poor risk-adjusted returns');
  } else if (sharpe < 2.0) {
    console.log('  1.0-2.0: Good risk-adjusted returns');
  } else if (sharpe < 3.0) {
    console.log('  2.0-3.0: Excellent risk-adjusted returns');
  } else {
    console.log('  > 3.0: Outstanding (verify data quality)');
  }

  console.log('');
  console.log(`Sortino/Sharpe Ratio: ${(sortino / sharpe).toFixed(2)}`);
  if (sortino / sharpe > 1.3) {
    console.log('  Returns have positive skew (good)');
  } else if (sortino / sharpe < 1.0) {
    console.log('  Returns have negative skew (concerning)');
  }

  // Calculate statistical metrics
  const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
  const stdDev = Math.sqrt(
    returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length
  );

  console.log('');
  console.log('Return Statistics:');
  console.log(`Average Daily Return: ${(avgReturn * 100).toFixed(3)}%`);
  console.log(`Daily Volatility: ${(stdDev * 100).toFixed(3)}%`);
  console.log(`Annualized Return: ${(avgReturn * 252 * 100).toFixed(2)}%`);
  console.log(`Annualized Volatility: ${(stdDev * Math.sqrt(252) * 100).toFixed(2)}%`);
}
```

### Leverage Management

```typescript
import { calculateMaxLeverage } from '@neural-trader/risk';

function determineLeverageLimits(
  portfolioValue: number,
  dailyVolatility: number,
  maxVolTarget: number = 0.20  // 20% annual vol target
) {
  const maxLeverage = calculateMaxLeverage(
    portfolioValue,
    dailyVolatility,
    maxVolTarget
  );

  const annualizedVol = dailyVolatility * Math.sqrt(252);

  console.log('Leverage Analysis:');
  console.log('='.repeat(50));
  console.log(`Portfolio Value: $${portfolioValue.toLocaleString()}`);
  console.log(`Daily Volatility: ${(dailyVolatility * 100).toFixed(2)}%`);
  console.log(`Annualized Volatility: ${(annualizedVol * 100).toFixed(2)}%`);
  console.log(`Target Volatility: ${(maxVolTarget * 100).toFixed(0)}%`);
  console.log('');
  console.log(`Maximum Recommended Leverage: ${maxLeverage.toFixed(2)}x`);
  console.log(`Maximum Position Size: $${(portfolioValue * maxLeverage).toLocaleString()}`);

  // Warnings
  if (maxLeverage < 1.0) {
    console.log('\n⚠️  Current volatility exceeds target.');
    console.log('Consider reducing exposure or hedging positions.');
  } else if (maxLeverage > 3.0) {
    console.log('\n⚠️  High leverage available but use with caution.');
    console.log('Higher leverage increases both gains and losses.');
  }

  return maxLeverage;
}

// Example: Determine safe leverage level
const maxLeverage = determineLeverageLimits(
  100000,   // $100k portfolio
  0.015,    // 1.5% daily volatility
  0.20      // 20% annual volatility target
);
```

## API Reference

### RiskManager

```typescript
class RiskManager {
  constructor(config: RiskConfig);

  // Value at Risk calculation
  calculateVar(returns: number[], portfolioValue: number): VaRResult;

  // Conditional Value at Risk
  calculateCvar(returns: number[], portfolioValue: number): CVaRResult;

  // Kelly Criterion position sizing
  calculateKelly(winRate: number, avgWin: number, avgLoss: number): KellyResult;

  // Drawdown metrics
  calculateDrawdown(equityCurve: number[]): DrawdownMetrics;

  // Risk-based position sizing
  calculatePositionSize(
    portfolioValue: number,
    pricePerShare: number,
    riskPerTrade: number,
    stopLossDistance: number
  ): PositionSize;

  // Validate position against limits
  validatePosition(
    positionSize: number,
    portfolioValue: number,
    maxPositionPercentage: number
  ): boolean;
}
```

### Utility Functions

```typescript
// Sharpe ratio calculation
function calculateSharpeRatio(
  returns: number[],
  riskFreeRate: number,
  annualizationFactor: number
): number;

// Sortino ratio calculation
function calculateSortinoRatio(
  returns: number[],
  targetReturn: number,
  annualizationFactor: number
): number;

// Maximum leverage calculation
function calculateMaxLeverage(
  portfolioValue: number,
  volatility: number,
  maxVolatilityTarget: number
): number;
```

## Configuration

### RiskConfig Options

| Option | Type | Description | Recommended |
|--------|------|-------------|-------------|
| `confidenceLevel` | `number` | Confidence level for VaR/CVaR | 0.95 (95%) |
| `lookbackPeriods` | `number` | Historical data periods | 252 (1 year daily) |
| `method` | `string` | VaR calculation method | 'historical' |

### VaR Methods

- **historical**: Non-parametric, uses actual return distribution
- **parametric**: Assumes normal distribution
- **monte-carlo**: Simulation-based (future release)

## Examples

### Example 1: Complete Risk Dashboard

```typescript
async function generateRiskDashboard(
  portfolio: any,
  returns: number[],
  equityCurve: number[]
) {
  const riskManager = new RiskManager({
    confidenceLevel: 0.95,
    lookbackPeriods: 252,
    method: 'historical'
  });

  console.log('RISK DASHBOARD');
  console.log('='.repeat(60));
  console.log('');

  // VaR Analysis
  const var95 = riskManager.calculateVar(returns, portfolio.value);
  const cvar95 = riskManager.calculateCvar(returns, portfolio.value);
  console.log('Value at Risk:');
  console.log(`  VaR (95%): $${var95.varAmount.toLocaleString()}`);
  console.log(`  CVaR (95%): $${cvar95.cvarAmount.toLocaleString()}`);
  console.log('');

  // Drawdown Analysis
  const dd = riskManager.calculateDrawdown(equityCurve);
  console.log('Drawdown Metrics:');
  console.log(`  Max Drawdown: ${(dd.maxDrawdown * 100).toFixed(2)}%`);
  console.log(`  Current Drawdown: ${(dd.currentDrawdown * 100).toFixed(2)}%`);
  console.log(`  Recovery Factor: ${dd.recoveryFactor.toFixed(2)}`);
  console.log('');

  // Risk-Adjusted Returns
  const sharpe = calculateSharpeRatio(returns, 0.02, 252);
  const sortino = calculateSortinoRatio(returns, 0.02 / 252, 252);
  console.log('Risk-Adjusted Performance:');
  console.log(`  Sharpe Ratio: ${sharpe.toFixed(2)}`);
  console.log(`  Sortino Ratio: ${sortino.toFixed(2)}`);
  console.log('');

  // Leverage Analysis
  const dailyVol = Math.sqrt(
    returns.reduce((sum, r) => sum + r * r, 0) / returns.length
  );
  const maxLev = calculateMaxLeverage(portfolio.value, dailyVol, 0.20);
  console.log('Leverage:');
  console.log(`  Current Volatility: ${(dailyVol * Math.sqrt(252) * 100).toFixed(2)}%`);
  console.log(`  Max Recommended Leverage: ${maxLev.toFixed(2)}x`);
}
```

## License

This package is dual-licensed under MIT OR Apache-2.0.

**MIT License**: https://opensource.org/licenses/MIT
**Apache-2.0 License**: https://www.apache.org/licenses/LICENSE-2.0

You may choose either license for your use.
