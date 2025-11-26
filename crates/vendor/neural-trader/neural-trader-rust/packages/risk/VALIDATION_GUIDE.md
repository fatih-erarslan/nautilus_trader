# Risk Package Validation Guide

This guide explains how to use the validation schemas in the `@neural-trader/risk` package.

## Overview

The risk package provides comprehensive risk management calculations with validated inputs. All financial data and calculations must pass validation before reaching the Rust layer.

## Available Validators

### Configuration
- `validateRiskConfig(config)` - Validates risk manager configuration

### Input Validators
- `validateReturns(returns)` - Validates return data
- `validateEquityCurve(curve)` - Validates equity curve
- `validateVarInput(input)` - Validates VAR calculation inputs
- `validateCvarInput(input)` - Validates CVaR calculation inputs
- `validateKellyInput(input)` - Validates Kelly Criterion inputs
- `validateDrawdownInput(input)` - Validates drawdown calculation inputs
- `validatePositionSizeInput(input)` - Validates position sizing inputs
- `validateSharpeRatioInput(input)` - Validates Sharpe Ratio inputs
- `validateSortinoRatioInput(input)` - Validates Sortino Ratio inputs
- `validateMaxLeverageInput(input)` - Validates max leverage inputs

## Risk Manager Configuration

```typescript
import { ValidatedRiskManager, ValidationError } from '@neural-trader/risk';
import type { RiskConfig } from '@neural-trader/core';

const config: RiskConfig = {
  confidenceLevel: 0.95,        // 95% confidence for VaR/CVaR
  riskFreeRate: 0.02,           // 2% risk-free rate
  lookbackPeriod: 252,          // 1 year of trading days
  maxDrawdown: 0.20,            // 20% max drawdown
  maxPositionSize: 0.10,        // 10% max per position
  maxLeverage: 2.0,             // 2x leverage maximum
  annualizationFactor: 252      // 252 trading days/year
};

try {
  const riskMgr = new ValidatedRiskManager(config);
  console.log('Risk manager initialized');
} catch (error) {
  if (error instanceof ValidationError) {
    console.error(`Configuration error: ${error.message}`);
  }
}
```

### Configuration Rules

- **confidenceLevel**: 0-1 (default: 0.95)
- **riskFreeRate**: 0-1 (default: 0.02)
- **lookbackPeriod**: 1-5000 days (default: 252)
- **maxDrawdown**: 0-1 (default: 0.1)
- **maxPositionSize**: 0-1 (default: 0.1)
- **maxLeverage**: 1-100 (default: 2)
- **annualizationFactor**: ≥ 1 (default: 252)

## Value at Risk (VaR)

```typescript
import { validateVarInput } from '@neural-trader/risk';

// Historical returns (daily returns)
const returns = [0.001, 0.0015, -0.0005, 0.002, -0.001, ...];
const portfolioValue = 1000000;  // $1 million

try {
  const varResult = riskMgr.calculateVar(returns, portfolioValue);
  console.log(`VaR at 95% confidence: $${varResult.value}`);
  console.log(`Max loss: ${varResult.percentage * 100}%`);
} catch (error) {
  if (error instanceof ValidationError) {
    console.error(`VaR calculation failed: ${error.message}`);
  }
}
```

### VaR Interpretation

- VaR at 95% = "With 95% confidence, we won't lose more than X in one day"
- Used for risk monitoring and capital allocation
- Fails to account for extreme tail events

## Conditional Value at Risk (CVaR)

```typescript
// CVaR addresses VaR shortcomings
const cvarResult = riskMgr.calculateCvar(returns, portfolioValue);
console.log(`CVaR at 95% confidence: $${cvarResult.value}`);
console.log(`Expected shortfall: $${cvarResult.expectedShortfall}`);
```

### CVaR vs VaR

- CVaR = Average loss in worst X% of scenarios
- More conservative than VaR
- Better for tail risk measurement
- Recommended for risk limits

## Kelly Criterion

```typescript
// Trading performance metrics
const winRate = 0.55;      // 55% of trades win
const avgWin = 1000;       // Average winning trade: $1,000
const avgLoss = 800;       // Average losing trade: $800

try {
  const kellyResult = riskMgr.calculateKelly(winRate, avgWin, avgLoss);
  console.log(`Optimal Kelly fraction: ${kellyResult.fraction}`);
  console.log(`Recommended position size: ${kellyResult.recommendedSize}`);
  console.log(`Growth rate: ${kellyResult.growthRate * 100}% per trade`);
} catch (error) {
  if (error instanceof ValidationError) {
    console.error(`Kelly calculation failed: ${error.message}`);
  }
}
```

### Kelly Criterion Validation

- **winRate**: 0 < rate < 1 (strictly between)
- **avgWin**: > 0, must be positive
- **avgLoss**: > 0, must be positive
- Prevents over-leveraging and betting ruin

### Kelly Fraction Interpretation

- Kelly fraction tells you optimal portfolio allocation
- E.g., 0.25 Kelly = allocate 25% of bankroll to strategy
- Full Kelly = maximum growth but high drawdown
- Practical: Use 0.25-0.5 Kelly for safety

## Drawdown Analysis

```typescript
// Portfolio equity curve over time
const equityCurve = [
  100000,  // Start
  105000,  // +5%
  103000,  // -2%
  110000,  // +7%
  100000,  // -9% (peak-to-trough = 10% drawdown)
  115000   // Recovery
];

try {
  const drawdown = riskMgr.calculateDrawdown(equityCurve);
  console.log({
    maxDrawdown: drawdown.maxDrawdown,
    maxDrawdownPercent: `${drawdown.maxDrawdownPercent * 100}%`,
    drawdownDuration: `${drawdown.drawdownDuration} days`,
    recoveryTime: `${drawdown.recoveryTime} days`,
    currentDrawdown: drawdown.currentDrawdown
  });
} catch (error) {
  if (error instanceof ValidationError) {
    console.error(`Drawdown calculation failed: ${error.message}`);
  }
}
```

### Drawdown Metrics

- **maxDrawdown**: Worst peak-to-trough loss
- **drawdownDuration**: Days spent in drawdown
- **recoveryTime**: Days to recover from max drawdown
- **currentDrawdown**: Current distance from peak

## Position Sizing

```typescript
const positionSizeInput = {
  portfolioValue: 100000,    // Total portfolio
  pricePerShare: 150,        // Stock price
  riskPerTrade: 0.02,        // Risk 2% of portfolio
  stopLossDistance: 5        // Stop loss 5 points away
};

try {
  const positionSize = riskMgr.calculatePositionSize(
    positionSizeInput.portfolioValue,
    positionSizeInput.pricePerShare,
    positionSizeInput.riskPerTrade,
    positionSizeInput.stopLossDistance
  );

  console.log({
    shares: positionSize.shares,
    dollarAmount: positionSize.dollarAmount,
    maxLoss: positionSize.maxLoss,
    riskReward: positionSize.riskReward
  });
} catch (error) {
  if (error instanceof ValidationError) {
    console.error(`Position sizing failed: ${error.message}`);
  }
}
```

### Position Sizing Formula

```
Position Size = (Account Risk $ / Stop Loss $) * Stock Price
Position Size = (Portfolio * Risk% / Stop Loss Distance) * Price
```

## Position Validation

```typescript
const positionSize = 50000;         // $50,000 position
const portfolioValue = 100000;      // $100,000 total
const maxPositionPercent = 0.10;    // Max 10% per position

try {
  const isValid = riskMgr.validatePosition(
    positionSize,
    portfolioValue,
    maxPositionPercent
  );

  if (isValid) {
    console.log('Position size is within limits');
  } else {
    console.log('Position exceeds maximum allowed size');
  }
} catch (error) {
  if (error instanceof ValidationError) {
    console.error(`Validation failed: ${error.message}`);
  }
}
```

## Sharpe Ratio

```typescript
import { calculateSharpeRatio } from '@neural-trader/risk';

const dailyReturns = [0.001, 0.0015, -0.0005, 0.002, ...];
const riskFreeRate = 0.02 / 252;  // Daily risk-free rate
const annualizationFactor = 252;   // Trading days per year

try {
  const sharpe = calculateSharpeRatio(dailyReturns, riskFreeRate, annualizationFactor);
  console.log(`Sharpe Ratio: ${sharpe.toFixed(2)}`);

  // Interpretation:
  // > 1.0 = Good
  // > 2.0 = Very Good
  // > 3.0 = Excellent
} catch (error) {
  if (error instanceof ValidationError) {
    console.error(`Sharpe calculation failed: ${error.message}`);
  }
}
```

### Sharpe Ratio Interpretation

- **Sharpe > 1.0**: Good risk-adjusted returns
- **Sharpe > 2.0**: Very good risk-adjusted returns
- **Sharpe > 3.0**: Excellent (rarely achieved)
- Compares excess returns to volatility

## Sortino Ratio

```typescript
import { calculateSortinoRatio } from '@neural-trader/risk';

const dailyReturns = [0.001, -0.002, 0.0015, 0.003, -0.0008, ...];
const targetReturn = 0;             // Minimize downside below 0
const annualizationFactor = 252;

try {
  const sortino = calculateSortinoRatio(dailyReturns, targetReturn, annualizationFactor);
  console.log(`Sortino Ratio: ${sortino.toFixed(2)}`);

  // Interpretation:
  // Similar to Sharpe, but only penalizes downside volatility
  // Usually higher than Sharpe Ratio
} catch (error) {
  if (error instanceof ValidationError) {
    console.error(`Sortino calculation failed: ${error.message}`);
  }
}
```

### Sortino Ratio Advantages

- Only penalizes downside volatility (below target)
- Upside volatility doesn't hurt the ratio
- Better for strategies with asymmetric returns
- Preferred for tail-risk focused strategies

## Maximum Leverage

```typescript
import { calculateMaxLeverage } from '@neural-trader/risk';

const portfolioValue = 100000;      // $100,000
const volatility = 0.20;            // 20% annual volatility
const maxVolatilityTarget = 0.15;   // Target 15% portfolio volatility

try {
  const maxLev = calculateMaxLeverage(portfolioValue, volatility, maxVolatilityTarget);
  console.log(`Maximum safe leverage: ${maxLev.toFixed(2)}x`);

  // E.g., 1.5x leverage = Use $150,000 buying power with $100k account
} catch (error) {
  if (error instanceof ValidationError) {
    console.error(`Leverage calculation failed: ${error.message}`);
  }
}
```

### Leverage Formula

```
Max Leverage = Target Volatility / Strategy Volatility
```

## Complete Risk Management Workflow

```typescript
import { ValidatedRiskManager, calculateSharpeRatio } from '@neural-trader/risk';

async function manageRisk() {
  try {
    // Initialize risk manager
    const riskMgr = new ValidatedRiskManager({
      confidenceLevel: 0.95,
      riskFreeRate: 0.02,
      maxDrawdown: 0.20
    });

    // Get trading performance data
    const returns = await getHistoricalReturns();  // Daily returns
    const equityCurve = await getEquityCurve();    // Portfolio value over time

    // Calculate key metrics
    const var95 = riskMgr.calculateVar(returns, 1000000);
    const cvar95 = riskMgr.calculateCvar(returns, 1000000);
    const sharpe = calculateSharpeRatio(returns, 0.02 / 252, 252);
    const drawdown = riskMgr.calculateDrawdown(equityCurve);

    // Check position sizing
    const posSize = riskMgr.calculatePositionSize(
      1000000,    // Portfolio value
      150,        // Stock price
      0.02,       // Risk 2% per trade
      5           // Stop loss 5 points away
    );

    // Validate current position
    const isValid = riskMgr.validatePosition(
      50000,      // Position size
      1000000,    // Portfolio value
      0.10        // Max 10% per position
    );

    // Report
    console.log({
      riskMetrics: {
        var95: var95.value,
        cvar95: cvar95.value,
        sharpeRatio: sharpe,
        maxDrawdown: drawdown.maxDrawdown,
        currentDrawdown: drawdown.currentDrawdown
      },
      positionSizing: {
        shares: posSize.shares,
        dollarAmount: posSize.dollarAmount,
        maxLoss: posSize.maxLoss
      },
      validation: {
        positionValid: isValid
      }
    });

  } catch (error) {
    if (error instanceof ValidationError) {
      console.error(`Risk management error: ${error.message}`);
    } else {
      console.error(`Unexpected error: ${error}`);
    }
  }
}
```

## Testing

Run validation tests:

```bash
npm run test:validation
```

Run all tests:

```bash
npm run test
```

## Best Practices

1. **Use appropriate confidence levels** - 95% for monitoring, 99% for limits
2. **Monitor multiple metrics** - Don't rely on VaR alone
3. **Use Kelly Criterion carefully** - Consider 0.25-0.5 Kelly for safety
4. **Track drawdown** - Monitor recovery time and duration
5. **Validate inputs** - Always pass clean, validated data
6. **Test on historical data** - Backtest risk calculations
7. **Update parameters regularly** - Use recent data for calculations
8. **Handle edge cases** - Be aware of model limitations

## Limitations to Consider

- Historical volatility may not predict future volatility
- VaR doesn't capture extreme tail events
- Correlations can break down during stress
- Kelly Criterion assumes fixed probabilities
- Drawdown recovery time is not linear

## See Also

- [`validation.ts`](./validation.ts) - Schema definitions
- [`validation.test.ts`](./validation.test.ts) - Test examples
- [`validation-wrapper.ts`](./validation-wrapper.ts) - Wrapper implementation
- [`README.md`](./README.md) - Package documentation
