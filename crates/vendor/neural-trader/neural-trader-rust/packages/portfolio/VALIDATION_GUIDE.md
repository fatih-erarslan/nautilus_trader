# Portfolio Package Validation Guide

This guide explains how to use the validation schemas in the `@neural-trader/portfolio` package.

## Overview

The portfolio package provides validated portfolio management and optimization. All positions and configurations must pass validation before reaching the Rust layer.

## Available Validators

### Portfolio Manager Validators

- `validatePortfolioConfig(config)` - Validates portfolio initialization
- `validatePositionUpdate(update)` - Validates position updates
- `validateSymbol(symbol)` - Validates symbol format

### Portfolio Optimizer Validators

- `validateOptimizerConfig(config)` - Validates optimizer configuration
- `validateOptimizationInput(input)` - Validates optimization input
- `validateRiskMetricsInput(input)` - Validates risk metrics calculation

## Portfolio Manager

### Initialization

```typescript
import { ValidatedPortfolioManager, ValidationError } from '@neural-trader/portfolio';

try {
  const initialCash = 100000;
  const portfolio = new ValidatedPortfolioManager(initialCash);
  console.log('Portfolio initialized with $100,000');
} catch (error) {
  if (error instanceof ValidationError) {
    console.error(`Portfolio initialization failed: ${error.message}`);
  }
}
```

### Position Management

#### Update Position

```typescript
try {
  // Buy 100 shares of AAPL at $150
  const position = await portfolio.updatePosition('AAPL', 100, 150.00);
  console.log(`Position updated:`, position);

  // Sell 50 shares of MSFT at $300
  const position2 = await portfolio.updatePosition('MSFT', -50, 300.00);
  console.log(`Position sold:`, position2);
} catch (error) {
  if (error instanceof ValidationError) {
    console.error(`Position update failed: ${error.message}`);
  }
}
```

#### Get Positions

```typescript
try {
  // Get all positions
  const allPositions = await portfolio.getPositions();
  console.log('All positions:', allPositions);

  // Get specific position
  const aaplPosition = await portfolio.getPosition('AAPL');
  if (aaplPosition) {
    console.log(`AAPL position: ${aaplPosition.quantity} @ $${aaplPosition.price}`);
  } else {
    console.log('No AAPL position');
  }
} catch (error) {
  console.error(`Failed to get positions: ${error.message}`);
}
```

#### Portfolio Metrics

```typescript
try {
  // Cash available
  const cash = await portfolio.getCash();
  console.log(`Available cash: $${cash}`);

  // Total portfolio value (positions + cash)
  const totalValue = await portfolio.getTotalValue();
  console.log(`Total portfolio value: $${totalValue}`);

  // Profit/Loss
  const pnl = await portfolio.getTotalPnl();
  console.log(`Total PnL: $${pnl}`);
} catch (error) {
  console.error(`Failed to get metrics: ${error.message}`);
}
```

### Position Validation Rules

- **Symbol**: Uppercase letters, numbers, hyphens (e.g., `AAPL`, `BRK-B`)
- **Quantity**: Non-zero integer (positive for long, negative for short)
- **Price**: Positive number
- **Entry Price** (optional): Positive number for tracking

## Portfolio Optimizer

### Basic Optimization

```typescript
import { ValidatedPortfolioOptimizer } from '@neural-trader/portfolio';
import type { OptimizerConfig } from '@neural-trader/core';

const optimizerConfig: OptimizerConfig = {
  method: 'markowitz'
};

try {
  const optimizer = new ValidatedPortfolioOptimizer(optimizerConfig);
  console.log('Optimizer initialized');
} catch (error) {
  if (error instanceof ValidationError) {
    console.error(`Optimizer initialization failed: ${error.message}`);
  }
}
```

### Mean-Variance Optimization

```typescript
// Historical returns for 3 assets (annual)
const returns = [0.10, 0.15, 0.08];

// Covariance matrix (annual)
const covariance = [
  [0.04, 0.006, 0.004],    // Asset 1 variance: 4%, cov with 2: 0.6%, cov with 3: 0.4%
  [0.006, 0.09, 0.008],    // Asset 2 variance: 9%, cov with 3: 0.8%
  [0.004, 0.008, 0.025]    // Asset 3 variance: 2.5%
];

const symbols = ['STOCK_A', 'STOCK_B', 'STOCK_C'];

try {
  const result = await optimizer.optimize(symbols, returns, covariance);
  console.log('Optimal allocation:', result);
  // Result includes weights, expected return, and expected volatility
} catch (error) {
  if (error instanceof ValidationError) {
    console.error(`Optimization failed: ${error.message}`);
  }
}
```

### Risk Parity Optimization

```typescript
const config: OptimizerConfig = {
  method: 'risk_parity',
  constraints: {
    minAllocation: 0.05,      // Min 5% per asset
    maxAllocation: 0.50       // Max 50% per asset
  }
};

const optimizer = new ValidatedPortfolioOptimizer(config);
const result = await optimizer.optimize(symbols, returns, covariance);
```

### Maximum Sharpe Ratio

```typescript
const config: OptimizerConfig = {
  method: 'max_sharpe',
  constraints: {
    targetVolatility: 0.15    // Target 15% volatility
  }
};

const optimizer = new ValidatedPortfolioOptimizer(config);
const result = await optimizer.optimize(symbols, returns, covariance);
```

### Minimum Variance

```typescript
const config: OptimizerConfig = {
  method: 'min_variance'
};

const optimizer = new ValidatedPortfolioOptimizer(config);
const result = await optimizer.optimize(symbols, returns, covariance);
```

### Risk Calculation

```typescript
const positions = {
  'AAPL': 25000,    // $25,000 in AAPL
  'MSFT': 30000,    // $30,000 in MSFT
  'GOOGL': 20000    // $20,000 in GOOGL
};

try {
  const riskMetrics = optimizer.calculateRisk(positions);
  console.log('Portfolio risk:', {
    totalValue: riskMetrics.total,
    volatility: riskMetrics.volatility,
    beta: riskMetrics.beta,
    concentration: riskMetrics.concentration
  });
} catch (error) {
  console.error(`Risk calculation failed: ${error.message}`);
}
```

## Optimizer Configuration Rules

### Optimization Methods
- `markowitz`: Classic Mean-Variance Optimization
- `risk_parity`: Equal risk contribution
- `min_variance`: Minimum portfolio variance
- `max_sharpe`: Maximum Sharpe Ratio

### Constraints
- **minAllocation**: 0-1 (default: 0)
- **maxAllocation**: 0-1 (default: 1)
- **targetVolatility**: Optional, 0+ value
- **minReturn**: Optional, target minimum return
- **leverageLimit**: 1-10 (default: unlimited)

### Rebalancing
- **rebalancingPeriod**: 1-365 days (optional)

## Optimization Input Validation Rules

- **Symbols**: 2-100 unique symbols, valid format
- **Returns**: Same count as symbols, finite numbers
- **Covariance**: Square matrix, matches symbol count
- **Correlations**: Values between -1 and 1 (implicit)

## Complete Workflow Example

```typescript
import {
  ValidatedPortfolioManager,
  ValidatedPortfolioOptimizer,
  ValidationError
} from '@neural-trader/portfolio';

async function managePortfolio() {
  try {
    // Initialize portfolio
    const portfolio = new ValidatedPortfolioManager(100000);

    // Build initial positions
    await portfolio.updatePosition('AAPL', 100, 150.00);   // $15,000
    await portfolio.updatePosition('MSFT', 150, 300.00);   // $45,000
    await portfolio.updatePosition('GOOGL', 10, 2800.00);  // $28,000

    // Check portfolio status
    const value = await portfolio.getTotalValue();
    const cash = await portfolio.getCash();
    const pnl = await portfolio.getTotalPnl();

    console.log(`Portfolio: Value=$${value}, Cash=$${cash}, PnL=$${pnl}`);

    // Historical returns and covariance (annual)
    const returns = [0.25, 0.20, 0.22];
    const covariance = [
      [0.08, 0.02, 0.018],
      [0.02, 0.06, 0.015],
      [0.018, 0.015, 0.07]
    ];

    // Optimize portfolio
    const optimizer = new ValidatedPortfolioOptimizer({
      method: 'max_sharpe',
      constraints: {
        minAllocation: 0.10,   // At least 10% per position
        maxAllocation: 0.50    // At most 50% per position
      }
    });

    const result = await optimizer.optimize(['AAPL', 'MSFT', 'GOOGL'], returns, covariance);
    console.log('Optimal allocation:', {
      weights: result.weights,
      expectedReturn: result.expectedReturn,
      expectedVolatility: result.expectedVolatility,
      sharpeRatio: result.sharpeRatio
    });

    // Calculate current risk
    const positions = {
      'AAPL': 15000,
      'MSFT': 45000,
      'GOOGL': 28000
    };

    const riskMetrics = optimizer.calculateRisk(positions);
    console.log('Portfolio risk metrics:', riskMetrics);

  } catch (error) {
    if (error instanceof ValidationError) {
      console.error(`Validation error: ${error.message}`);
    } else {
      console.error(`Error: ${error}`);
    }
  }
}

managePortfolio();
```

## Error Handling

```typescript
import { ValidationError } from '@neural-trader/portfolio';

try {
  // Try invalid operation
  await portfolio.updatePosition('aapl', 100, 150);  // lowercase symbol
} catch (error) {
  if (error instanceof ValidationError) {
    console.error(`Validation failed: ${error.message}`);
    console.error(`Original error:`, error.originalError);
  } else {
    console.error(`Unexpected error: ${error}`);
  }
}
```

## Testing

Run validation tests:

```bash
npm run test:validation
```

Test specific validator:

```bash
npm run test -- --testNamePattern="symbolSchema"
```

Run all tests:

```bash
npm run test
```

## Best Practices

1. **Validate symbols** - Always use uppercase format
2. **Check available cash** - Ensure sufficient funds before trading
3. **Monitor positions** - Regularly check portfolio composition
4. **Use realistic inputs** - Base returns/covariance on actual data
5. **Rebalance periodically** - Follow optimization recommendations
6. **Handle errors** - Always catch ValidationError
7. **Test optimization** - Verify results with historical data

## Performance Tips

- Cache covariance matrices for repeated calculations
- Reuse optimizer instances for similar calculations
- Batch position updates when possible
- Monitor portfolio metrics asynchronously

## See Also

- [`validation.ts`](./validation.ts) - Schema definitions
- [`validation.test.ts`](./validation.test.ts) - Test examples
- [`validation-wrapper.ts`](./validation-wrapper.ts) - Wrapper implementation
- [`README.md`](./README.md) - Package documentation
