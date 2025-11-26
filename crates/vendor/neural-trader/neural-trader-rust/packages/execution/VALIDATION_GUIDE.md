# Execution Package Validation Guide

This guide explains how to use the validation schemas in the `@neural-trader/execution` package.

## Overview

The execution package provides validated order placement and account management. All trading orders and configuration must pass validation before reaching the Rust layer.

## Available Validators

### Core Schemas

- `executionConfigSchema` - Validates broker configuration
- `orderSchema` - Validates individual orders
- `batchOrderSchema` - Validates multiple orders
- `orderUpdateSchema` - Validates order updates

### Helper Functions

- `validateExecutionConfig(config)` - Validates execution configuration
- `validateOrder(order)` - Validates a single order
- `validateBatchOrders(orders)` - Validates order batch
- `validateOrderUpdate(update)` - Validates order update

## Configuration

### Execution Configuration

```typescript
import { ValidatedNeuralTrader, ValidationError } from '@neural-trader/execution';
import type { JsConfig } from '@neural-trader/core';

const config: JsConfig = {
  brokerId: 'INTERACTIVE_BROKERS',
  apiKey: process.env.IB_API_KEY,
  apiSecret: process.env.IB_API_SECRET,
  accountId: 'ACC123456',
  maxSlippage: 0.001,        // 0.1% max slippage
  timeout: 5000,             // 5 second timeout
  orderRetryCount: 3         // Retry failed orders 3 times
};

try {
  const trader = new ValidatedNeuralTrader(config);
  console.log('Trader initialized and validated');
} catch (error) {
  if (error instanceof ValidationError) {
    console.error(`Configuration error: ${error.message}`);
  }
}
```

### Configuration Validation Rules

- **brokerId**: Required, 1-100 characters
- **apiKey**: Required, 1-500 characters
- **apiSecret**: Required, 1-500 characters, never logged
- **endpoint** (optional): Must be valid URL
- **accountId** (optional): 1-100 characters
- **maxSlippage**: 0-1 (default: 0.001)
- **timeout**: 100-60000 ms (default: 5000)
- **orderRetryCount**: 0-10 (default: 3)

## Order Placement

### Basic Market Order

```typescript
import type { JsOrder } from '@neural-trader/core';

const trader = new ValidatedNeuralTrader(config);
await trader.start();

const order: JsOrder = {
  symbol: 'AAPL',
  side: 'BUY',
  quantity: 100,
  price: 150.50,
  orderType: 'MARKET'
};

try {
  const result = await trader.placeOrder(order);
  console.log('Market order placed:', result);
} catch (error) {
  if (error instanceof ValidationError) {
    console.error(`Order validation failed: ${error.message}`);
  }
}
```

### Limit Order

```typescript
const limitOrder: JsOrder = {
  symbol: 'MSFT',
  side: 'BUY',
  quantity: 50,
  price: 300.00,
  orderType: 'LIMIT',
  timeInForce: 'GTC'  // Good-Till-Cancelled
};

const result = await trader.placeOrder(limitOrder);
```

### Stop Order

```typescript
const stopOrder: JsOrder = {
  symbol: 'GOOGL',
  side: 'SELL',
  quantity: 25,
  price: 2800.00,
  orderType: 'STOP',
  stopPrice: 2750.00,
  timeInForce: 'GTC'
};

const result = await trader.placeOrder(stopOrder);
```

### Advanced Execution Strategies

#### TWAP (Time-Weighted Average Price)

```typescript
const twapOrder: JsOrder = {
  symbol: 'SPY',
  side: 'BUY',
  quantity: 10000,
  price: 400.00,
  executionStrategy: 'TWAP',
  sliceDuration: 3600000,  // 1 hour
  sliceCount: 12           // 12 equal slices
};

await trader.placeOrder(twapOrder);
```

#### VWAP (Volume-Weighted Average Price)

```typescript
const vwapOrder: JsOrder = {
  symbol: 'QQQ',
  side: 'SELL',
  quantity: 5000,
  price: 350.00,
  executionStrategy: 'VWAP',
  sliceDuration: 1800000,  // 30 minutes
  sliceCount: 6
};

await trader.placeOrder(vwapOrder);
```

#### Iceberg Order

```typescript
const icebergOrder: JsOrder = {
  symbol: 'BRK.B',
  side: 'BUY',
  quantity: 5000,
  price: 380.00,
  executionStrategy: 'ICEBERG',
  icebergQty: 100         // Show 100 at a time
};

await trader.placeOrder(icebergOrder);
```

### Batch Order Placement

```typescript
const orders: JsOrder[] = [
  {
    symbol: 'AAPL',
    side: 'BUY',
    quantity: 100,
    price: 150.00
  },
  {
    symbol: 'MSFT',
    side: 'BUY',
    quantity: 50,
    price: 300.00
  },
  {
    symbol: 'GOOGL',
    side: 'SELL',
    quantity: 25,
    price: 2800.00
  }
];

try {
  const results = await trader.placeBatchOrders(orders);
  console.log('All orders placed:', results);
} catch (error) {
  if (error instanceof ValidationError) {
    console.error(`Batch order failed: ${error.message}`);
  }
}
```

## Order Validation Rules

### Symbol
- Format: Uppercase letters, numbers, hyphens (e.g., `AAPL`, `BRK-B`)
- 1-10 characters

### Side
- Must be `BUY` or `SELL`

### Quantity
- Must be positive number
- No fractional shares (for most brokers)

### Price
- Must be positive number
- Cannot be zero

### Order Type
- `MARKET`: No stop price required
- `LIMIT`: Standard limit order
- `STOP`: Requires `stopPrice`
- `STOP_LIMIT`: Requires `stopPrice` and limit `price`

### Time in Force
- `GTC`: Good-Till-Cancelled (default)
- `IOC`: Immediate-Or-Cancel
- `FOK`: Fill-Or-Kill
- `DAY`: Good for today only

### Execution Strategy
- `MARKET`: Immediate execution (default)
- `TWAP`: Time-weighted execution
- `VWAP`: Volume-weighted execution
- `ICEBERG`: Large order hidden as smaller visible orders
- `POV`: Percentage-of-volume execution

## Trading Workflow

```typescript
const trader = new ValidatedNeuralTrader(config);

try {
  // Start trading engine
  await trader.start();
  console.log('Trader started');

  // Place orders
  const order = { symbol: 'AAPL', side: 'BUY', quantity: 100, price: 150 };
  await trader.placeOrder(order);

  // Monitor positions
  const positions = await trader.getPositions();
  console.log('Current positions:', positions);

  // Check balance
  const balance = await trader.getBalance();
  console.log('Available cash:', balance);

  // Check equity
  const equity = await trader.getEquity();
  console.log('Total equity:', equity);

  // Stop trading engine
  await trader.stop();
  console.log('Trader stopped');
} catch (error) {
  console.error('Trading error:', error);
  if (trader.isRunning()) {
    await trader.stop();
  }
}
```

## Error Handling

```typescript
import { ValidationError } from '@neural-trader/execution';

try {
  const result = await trader.placeOrder(order);
} catch (error) {
  if (error instanceof ValidationError) {
    // Validation error - bad input parameters
    console.error(`Invalid order: ${error.message}`);
    console.error(`Details:`, error.originalError);
  } else if (error instanceof Error) {
    // Other errors (network, connection, etc.)
    console.error(`Trading error: ${error.message}`);
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

## Environment Variables

```bash
# Set API credentials
export IB_API_KEY=your_api_key
export IB_API_SECRET=your_api_secret

# Optional: Set custom endpoint
export BROKER_ENDPOINT=https://custom-broker.api.com
```

## Best Practices

1. **Always validate configuration** - Never hardcode credentials
2. **Use environment variables** - Store secrets securely
3. **Check trader status** - Verify `isRunning()` before operations
4. **Handle errors gracefully** - Catch and log all errors
5. **Clean up resources** - Always call `stop()` when done
6. **Use batch orders for efficiency** - Group related orders
7. **Test in sandbox** - Validate logic before live trading

## Safety Features

- All orders are validated before execution
- Connection timeout prevents hanging requests
- Automatic retry on transient failures
- Max slippage enforcement to prevent slippage abuse
- Account ID validation for multi-account trading

## See Also

- [`validation.ts`](./validation.ts) - Schema definitions
- [`validation.test.ts`](./validation.test.ts) - Test examples
- [`validation-wrapper.ts`](./validation-wrapper.ts) - Wrapper implementation
- [`README.md`](./README.md) - Package documentation
