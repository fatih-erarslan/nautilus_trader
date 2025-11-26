# Strategies Package Validation Guide

This guide explains how to use the validation schemas in the `@neural-trader/strategies` package.

## Overview

The validation system provides runtime validation of all input parameters using Zod schemas. This prevents invalid data from reaching the Rust layer and provides clear, actionable error messages.

## Installation

The validation dependencies are already included in `package.json`:

```bash
npm install
# zod is included as a dependency
```

## Available Validators

### Core Schemas

- `strategyConfigSchema` - Validates base strategy configuration
- `momentumParametersSchema` - Validates momentum strategy parameters
- `meanReversionParametersSchema` - Validates mean reversion strategy parameters
- `arbitrageParametersSchema` - Validates arbitrage strategy parameters
- `pairsTradingParametersSchema` - Validates pairs trading strategy parameters
- `strategyIdSchema` - Validates strategy ID format
- `signalCallbackSchema` - Validates signal callback function

### Helper Functions

All validators have corresponding helper functions that throw `ValidationError`:

- `validateStrategyConfig(config)` - Validates strategy configuration
- `validateMomentumParameters(params)` - Validates momentum parameters
- `validateMeanReversionParameters(params)` - Validates mean reversion parameters
- `validateArbitrageParameters(params)` - Validates arbitrage parameters
- `validatePairsTradingParameters(params)` - Validates pairs trading parameters
- `validateStrategyId(id)` - Validates strategy ID

## Usage Examples

### Basic Strategy Configuration

```typescript
import { ValidatedStrategyRunner, ValidationError } from '@neural-trader/strategies';
import type { StrategyConfig } from '@neural-trader/core';

const runner = new ValidatedStrategyRunner();

const config: StrategyConfig = {
  name: 'SMA Crossover',
  symbols: ['AAPL', 'MSFT', 'GOOGL'],
  parameters: JSON.stringify({
    shortPeriod: 20,
    longPeriod: 50,
    minVolume: 1000000
  })
};

try {
  const strategyId = await runner.addMomentumStrategy(config);
  console.log(`Strategy created: ${strategyId}`);
} catch (error) {
  if (error instanceof ValidationError) {
    console.error(`Validation failed: ${error.message}`);
  }
}
```

### Momentum Strategy

```typescript
import { validateMomentumParameters, ValidationError } from '@neural-trader/strategies';

const params = {
  shortPeriod: 20,
  longPeriod: 50,
  minVolume: 1000000
};

try {
  const validated = validateMomentumParameters(params);
  console.log('Parameters are valid');
} catch (error) {
  if (error instanceof ValidationError) {
    console.error(`Invalid parameters: ${error.message}`);
  }
}
```

### Mean Reversion Strategy

```typescript
const params = {
  period: 20,
  stdDevThreshold: 2.0,
  rsiThreshold: 30
};

const config: StrategyConfig = {
  name: 'Bollinger Bands',
  symbols: ['SPY', 'QQQ'],
  parameters: JSON.stringify(params)
};

try {
  const strategyId = await runner.addMeanReversionStrategy(config);
  console.log(`Mean reversion strategy added: ${strategyId}`);
} catch (error) {
  console.error(`Error: ${error}`);
}
```

### Arbitrage Strategy

```typescript
const params = {
  priceThreshold: 0.01,
  minProfit: 100,
  exchangePairs: [
    { exchange1: 'BINANCE', exchange2: 'KRAKEN' },
    { exchange1: 'COINBASE', exchange2: 'BINANCE' }
  ]
};

const config: StrategyConfig = {
  name: 'Cross-Exchange Arbitrage',
  symbols: ['BTC', 'ETH'],
  parameters: JSON.stringify(params)
};

const strategyId = await runner.addArbitrageStrategy(config);
```

### Pairs Trading Strategy

```typescript
const params = {
  symbol1: 'EWA',
  symbol2: 'EWC',
  cointegrationThreshold: 0.95,
  spreadThreshold: 5.0
};

const config: StrategyConfig = {
  name: 'Statistical Arbitrage',
  symbols: ['EWA', 'EWC'],
  parameters: JSON.stringify(params)
};

const strategyId = await runner.addPairsTradingStrategy(config);
```

## Validation Rules

### Strategy Name
- Must not be empty
- Maximum 255 characters

### Symbols
- Format: Uppercase letters, numbers, hyphens only (e.g., `AAPL`, `BRK-B`)
- Minimum 1, Maximum 100 per strategy
- Maximum 10 characters each

### Momentum Strategy Parameters
- `shortPeriod`: 1-200, integer
- `longPeriod`: 2-500, integer
- **Constraint**: `shortPeriod < longPeriod`
- `minVolume` (optional): >= 0

### Mean Reversion Parameters
- `period`: 1-500, integer
- `stdDevThreshold`: 0-5, positive number
- `rsiThreshold` (optional): 0-100

### Arbitrage Parameters
- `priceThreshold`: 0-1 (0-100%)
- `minProfit`: >= 0
- `exchangePairs`: Array with at least 1 pair

### Pairs Trading Parameters
- `symbol1` & `symbol2`: Valid symbol format, not empty
- `cointegrationThreshold`: 0-1
- `spreadThreshold`: >= 0

## Error Handling

The wrapper classes throw `ValidationError` for validation failures:

```typescript
import { ValidationError } from '@neural-trader/strategies';

try {
  // Attempt to add strategy
  const id = await runner.addMomentumStrategy(config);
} catch (error) {
  if (error instanceof ValidationError) {
    // Handle validation error
    console.error(`Field validation failed: ${error.message}`);
    console.error(`Original error:`, error.originalError);
  } else {
    // Handle other errors
    console.error('Unexpected error:', error);
  }
}
```

## Testing

Run the validation tests:

```bash
npm run test:validation
```

Run all tests:

```bash
npm run test
```

## Integration with Native Layer

The `ValidatedStrategyRunner` class wraps the native `StrategyRunner`:

1. Validates input parameters using Zod schemas
2. Provides clear error messages for invalid inputs
3. Delegates to native Rust layer only if validation passes
4. Maintains complete type safety with TypeScript

This ensures:
- Invalid data never reaches Rust code
- Performance is not impacted by validation (only at entry points)
- Clear error messages for debugging
- Automatic type inference in TypeScript

## Schema Exports

You can import and extend schemas as needed:

```typescript
import { strategyConfigSchema, momentumParametersSchema } from '@neural-trader/strategies';
import { z } from 'zod';

// Create a custom schema combining multiple validations
const extendedConfig = strategyConfigSchema.extend({
  customField: z.string().optional()
});
```

## Best Practices

1. **Always use ValidatedStrategyRunner** - Never use the native `StrategyRunner` directly
2. **Parse parameters as JSON** - Strategy parameters must be JSON strings
3. **Handle ValidationError** - Always catch and handle validation errors
4. **Test validation** - Use the test suite to verify your parameter configurations
5. **Reference documentation** - Keep parameter constraints documented in your code

## Performance Considerations

- Validation occurs only at entry points (minimal overhead)
- Schemas are compiled once and reused
- Zod provides optimized validation
- Invalid inputs fail fast with clear messages

## Migration Guide

If using the old native `StrategyRunner`:

```typescript
// Old way (no validation)
import { StrategyRunner } from '@neural-trader/strategies';
const runner = new StrategyRunner();

// New way (with validation)
import { ValidatedStrategyRunner } from '@neural-trader/strategies';
const runner = new ValidatedStrategyRunner();

// Usage is identical, but with validation
const id = await runner.addMomentumStrategy(config);
```

## Support

For validation schema issues or questions:
1. Check the schema definition in `validation.ts`
2. Run the validation tests: `npm run test:validation`
3. Review error messages - they clearly indicate what's wrong
4. Consult this guide and code examples

## See Also

- [`validation.ts`](./validation.ts) - Schema definitions
- [`validation.test.ts`](./validation.test.ts) - Test cases and examples
- [`validation-wrapper.ts`](./validation-wrapper.ts) - Wrapper implementation
