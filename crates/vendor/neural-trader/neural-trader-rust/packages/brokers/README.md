# @neural-trader/brokers

[![npm version](https://img.shields.io/npm/v/@neural-trader/brokers.svg)](https://www.npmjs.com/package/@neural-trader/brokers)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/ruvnet/neural-trader)
[![TypeScript](https://img.shields.io/badge/TypeScript-Ready-blue.svg)](https://www.typescriptlang.org/)

**Enterprise-grade broker integrations for Neural Trader** with unified API supporting Alpaca, Interactive Brokers, Binance, and Coinbase. Built with Rust for maximum performance and reliability.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Brokers](#supported-brokers)
- [Core Concepts](#core-concepts)
- [Examples](#examples)
- [Integration Patterns](#integration-patterns)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)
- [License](#license)

## Features

- **Multi-Broker Support**: Alpaca, Interactive Brokers, Binance, Coinbase with unified interface
- **Order Management**: Market, limit, stop, stop-limit orders with advanced time-in-force options
- **Account Management**: Real-time balance, positions, and portfolio tracking
- **Paper Trading**: Full-featured simulation mode for risk-free strategy testing
- **WebSocket Streaming**: Real-time order updates and execution notifications
- **Crypto & Equities**: Support for both traditional and digital asset trading
- **Rust Performance**: Lightning-fast order execution with sub-millisecond latency
- **Type Safety**: Full TypeScript definitions for all APIs
- **Error Recovery**: Automatic retry logic and connection recovery
- **Rate Limiting**: Built-in rate limit management per broker requirements

## Installation

```bash
# Install with required dependencies
npm install @neural-trader/brokers @neural-trader/core

# Or with yarn
yarn add @neural-trader/brokers @neural-trader/core

# Install type definitions
npm install --save-dev @types/node
```

### Dependencies

This package requires:
- `@neural-trader/core` - Core functionality and types
- `@neural-trader/execution` - Order execution engine (installed automatically)

## Quick Start

### Basic Usage

```typescript
import { BrokerClient, listBrokerTypes } from '@neural-trader/brokers';

// List available brokers
console.log('Available brokers:', listBrokerTypes());
// Output: ['alpaca', 'interactive_brokers', 'binance', 'coinbase']

// Connect to Alpaca for equities trading
const client = new BrokerClient({
  brokerType: 'alpaca',
  apiKey: process.env.ALPACA_API_KEY,
  apiSecret: process.env.ALPACA_SECRET,
  baseUrl: 'https://paper-api.alpaca.markets',
  paperTrading: true
});

await client.connect();
console.log('Connected to broker');

// Place a market order
const order = await client.placeOrder({
  symbol: 'AAPL',
  side: 'buy',
  orderType: 'market',
  quantity: 100,
  timeInForce: 'day'
});

console.log(`Order placed: ${order.orderId}`);
console.log(`Status: ${order.status}`);

// Check account balance
const balance = await client.getAccountBalance();
console.log(`Cash: $${balance.cash.toFixed(2)}`);
console.log(`Equity: $${balance.equity.toFixed(2)}`);

// Cleanup
await client.disconnect();
```

## Supported Brokers

### Alpaca
- **Asset Types**: US Equities, ETFs, Crypto (via Alpaca Crypto)
- **Paper Trading**: Yes (full feature parity)
- **API Base**: `https://paper-api.alpaca.markets` (paper), `https://api.alpaca.markets` (live)
- **Authentication**: API Key + Secret

### Interactive Brokers
- **Asset Types**: Stocks, Options, Futures, Forex, Bonds
- **Paper Trading**: Yes (IB Gateway required)
- **API Base**: Local gateway connection (port 4001/7496)
- **Authentication**: Account credentials

### Binance
- **Asset Types**: Cryptocurrencies (500+ pairs)
- **Paper Trading**: Testnet available
- **API Base**: `https://testnet.binance.vision` (testnet), `https://api.binance.com` (live)
- **Authentication**: API Key + Secret

### Coinbase
- **Asset Types**: Cryptocurrencies (200+ assets)
- **Paper Trading**: Sandbox available
- **API Base**: `https://api-public.sandbox.pro.coinbase.com` (sandbox), `https://api.coinbase.com` (live)
- **Authentication**: API Key + Secret + Passphrase

## Core Concepts

### Order Types

```typescript
// Market Order - Execute immediately at best available price
await client.placeOrder({
  symbol: 'AAPL',
  side: 'buy',
  orderType: 'market',
  quantity: 100,
  timeInForce: 'day'
});

// Limit Order - Execute only at specified price or better
await client.placeOrder({
  symbol: 'AAPL',
  side: 'sell',
  orderType: 'limit',
  quantity: 100,
  limitPrice: 175.50,
  timeInForce: 'gtc'
});

// Stop Order - Trigger market order when price reaches stop price
await client.placeOrder({
  symbol: 'AAPL',
  side: 'sell',
  orderType: 'stop',
  quantity: 100,
  stopPrice: 165.00,
  timeInForce: 'day'
});

// Stop-Limit Order - Trigger limit order at stop price
await client.placeOrder({
  symbol: 'AAPL',
  side: 'sell',
  orderType: 'stop_limit',
  quantity: 100,
  stopPrice: 165.00,
  limitPrice: 164.50,
  timeInForce: 'day'
});
```

### Time In Force Options

- `day` - Order valid until market close
- `gtc` - Good 'til canceled (90 days max)
- `ioc` - Immediate or cancel
- `fok` - Fill or kill (entire order must execute)
- `opg` - Execute at market open
- `cls` - Execute at market close

### Account Information

```typescript
// Get detailed account information
const balance = await client.getAccountBalance();
console.log('Account Details:');
console.log(`  Cash: $${balance.cash}`);
console.log(`  Equity: $${balance.equity}`);
console.log(`  Buying Power: $${balance.buyingPower}`);
console.log(`  Margin Used: $${balance.marginUsed || 0}`);

// Get all open positions
const positions = await client.getPositions();
positions.forEach(pos => {
  console.log(`${pos.symbol}: ${pos.quantity} shares @ $${pos.entryPrice}`);
  console.log(`  Current P&L: $${pos.unrealizedPl}`);
});
```

## Examples

### Cryptocurrency Trading (Binance)

```typescript
import { BrokerClient } from '@neural-trader/brokers';

const binance = new BrokerClient({
  brokerType: 'binance',
  apiKey: process.env.BINANCE_API_KEY,
  apiSecret: process.env.BINANCE_SECRET,
  baseUrl: 'https://testnet.binance.vision',
  paperTrading: true
});

await binance.connect();

// Place crypto order (BTC/USDT)
const btcOrder = await binance.placeOrder({
  symbol: 'BTCUSDT',
  side: 'buy',
  orderType: 'limit',
  quantity: 0.01,
  limitPrice: 42000.00,
  timeInForce: 'gtc'
});

console.log(`BTC Order ID: ${btcOrder.orderId}`);

// Monitor order status
const status = await binance.getOrderStatus(btcOrder.orderId);
console.log(`Order Status: ${status.status}`);
console.log(`Filled: ${status.filledQuantity}/${status.quantity}`);

await binance.disconnect();
```

### Multi-Broker Portfolio Management

```typescript
import { BrokerClient } from '@neural-trader/brokers';

// Manage positions across multiple brokers
const brokers = [
  new BrokerClient({ brokerType: 'alpaca', /* ... */ }),
  new BrokerClient({ brokerType: 'binance', /* ... */ }),
  new BrokerClient({ brokerType: 'coinbase', /* ... */ })
];

// Connect to all brokers
await Promise.all(brokers.map(b => b.connect()));

// Get consolidated portfolio view
const portfolios = await Promise.all(
  brokers.map(async broker => ({
    broker: broker.config.brokerType,
    balance: await broker.getAccountBalance(),
    positions: await broker.getPositions()
  }))
);

// Calculate total portfolio value
const totalEquity = portfolios.reduce(
  (sum, p) => sum + p.balance.equity,
  0
);
console.log(`Total Portfolio Value: $${totalEquity.toFixed(2)}`);

// Cleanup
await Promise.all(brokers.map(b => b.disconnect()));
```

### Order Management with Error Handling

```typescript
import { BrokerClient } from '@neural-trader/brokers';

async function placeOrderWithRetry(
  client: BrokerClient,
  order: OrderRequest,
  maxRetries = 3
): Promise<OrderResponse> {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await client.placeOrder(order);
    } catch (error) {
      console.error(`Order attempt ${attempt} failed:`, error.message);

      if (attempt === maxRetries) {
        throw new Error(`Failed to place order after ${maxRetries} attempts`);
      }

      // Wait before retry (exponential backoff)
      await new Promise(resolve =>
        setTimeout(resolve, Math.pow(2, attempt) * 1000)
      );
    }
  }
}

// Usage
const client = new BrokerClient({ /* config */ });
await client.connect();

try {
  const order = await placeOrderWithRetry(client, {
    symbol: 'AAPL',
    side: 'buy',
    orderType: 'limit',
    quantity: 100,
    limitPrice: 170.00,
    timeInForce: 'day'
  });

  console.log(`Order placed successfully: ${order.orderId}`);
} catch (error) {
  console.error('Failed to place order:', error);
}

await client.disconnect();
```

### Advanced Order Monitoring

```typescript
import { BrokerClient } from '@neural-trader/brokers';

const client = new BrokerClient({ /* config */ });
await client.connect();

// Place order
const order = await client.placeOrder({
  symbol: 'TSLA',
  side: 'buy',
  orderType: 'limit',
  quantity: 50,
  limitPrice: 250.00,
  timeInForce: 'day'
});

// Monitor order until filled or cancelled
async function monitorOrder(orderId: string): Promise<void> {
  const checkInterval = 1000; // 1 second
  const maxWaitTime = 300000; // 5 minutes
  const startTime = Date.now();

  while (Date.now() - startTime < maxWaitTime) {
    const status = await client.getOrderStatus(orderId);

    console.log(`Order ${orderId}:`);
    console.log(`  Status: ${status.status}`);
    console.log(`  Filled: ${status.filledQuantity}/${status.quantity}`);
    console.log(`  Avg Fill Price: $${status.averagePrice || 'N/A'}`);

    if (status.status === 'filled') {
      console.log('Order filled successfully!');
      return;
    }

    if (status.status === 'cancelled' || status.status === 'rejected') {
      console.log(`Order ${status.status}`);
      return;
    }

    await new Promise(resolve => setTimeout(resolve, checkInterval));
  }

  console.log('Order monitoring timeout - cancelling order');
  await client.cancelOrder(orderId);
}

await monitorOrder(order.orderId);
await client.disconnect();
```

## Integration Patterns

### Integration with Strategies

```typescript
import { BrokerClient } from '@neural-trader/brokers';
import { Strategy } from '@neural-trader/strategies';
import { MarketDataProvider } from '@neural-trader/market-data';

// Create trading strategy
const strategy = new Strategy({
  name: 'momentum-strategy',
  symbols: ['AAPL', 'MSFT', 'GOOGL'],
  parameters: {
    rsiPeriod: 14,
    rsiOverbought: 70,
    rsiOversold: 30
  }
});

// Connect broker and data provider
const broker = new BrokerClient({
  brokerType: 'alpaca',
  apiKey: process.env.ALPACA_API_KEY,
  apiSecret: process.env.ALPACA_SECRET,
  paperTrading: true
});

const dataProvider = new MarketDataProvider({
  provider: 'alpaca',
  apiKey: process.env.ALPACA_API_KEY,
  apiSecret: process.env.ALPACA_SECRET
});

await broker.connect();
await dataProvider.connect();

// Execute strategy signals through broker
strategy.on('signal', async (signal) => {
  console.log(`Signal: ${signal.action} ${signal.symbol}`);

  try {
    if (signal.action === 'buy') {
      const order = await broker.placeOrder({
        symbol: signal.symbol,
        side: 'buy',
        orderType: 'market',
        quantity: signal.quantity,
        timeInForce: 'day'
      });
      console.log(`Buy order placed: ${order.orderId}`);
    } else if (signal.action === 'sell') {
      const order = await broker.placeOrder({
        symbol: signal.symbol,
        side: 'sell',
        orderType: 'market',
        quantity: signal.quantity,
        timeInForce: 'day'
      });
      console.log(`Sell order placed: ${order.orderId}`);
    }
  } catch (error) {
    console.error(`Failed to execute signal:`, error);
  }
});

// Start strategy with real-time data
await strategy.start(dataProvider);
```

### Integration with Backtesting

```typescript
import { BrokerClient, validateBrokerConfig } from '@neural-trader/brokers';
import { BacktestEngine } from '@neural-trader/backtesting';

// Create backtest engine with broker simulation
const backtest = new BacktestEngine({
  initialCapital: 100000,
  commission: 0.001, // 0.1% per trade
  slippage: 0.0005,  // 0.05% slippage
  startDate: '2023-01-01',
  endDate: '2023-12-31'
});

// Define strategy that uses broker API patterns
backtest.addStrategy(async (context) => {
  const { data, portfolio } = context;

  // Simulate broker order placement
  if (data.rsi < 30) {
    // Buy signal
    const quantity = Math.floor(portfolio.cash / data.close / 2);
    if (quantity > 0) {
      await context.placeOrder({
        symbol: data.symbol,
        side: 'buy',
        orderType: 'market',
        quantity: quantity,
        timeInForce: 'day'
      });
    }
  } else if (data.rsi > 70) {
    // Sell signal
    const position = portfolio.positions.get(data.symbol);
    if (position && position.quantity > 0) {
      await context.placeOrder({
        symbol: data.symbol,
        side: 'sell',
        orderType: 'market',
        quantity: position.quantity,
        timeInForce: 'day'
      });
    }
  }
});

// Run backtest
const results = await backtest.run(['AAPL', 'MSFT']);

console.log('Backtest Results:');
console.log(`  Total Return: ${(results.totalReturn * 100).toFixed(2)}%`);
console.log(`  Sharpe Ratio: ${results.sharpeRatio.toFixed(2)}`);
console.log(`  Max Drawdown: ${(results.maxDrawdown * 100).toFixed(2)}%`);
console.log(`  Win Rate: ${(results.winRate * 100).toFixed(2)}%`);

// After successful backtest, deploy to live broker
if (results.sharpeRatio > 2.0 && results.maxDrawdown < 0.2) {
  console.log('Strategy passed validation - deploying to live trading');

  const broker = new BrokerClient({
    brokerType: 'alpaca',
    apiKey: process.env.ALPACA_API_KEY,
    apiSecret: process.env.ALPACA_SECRET,
    paperTrading: true // Start with paper trading
  });

  await broker.connect();
  console.log('Live trading initialized');
}
```

## API Reference

### BrokerClient

Main class for interacting with brokers.

```typescript
class BrokerClient {
  constructor(config: BrokerConfig);

  // Connection Management
  connect(): Promise<boolean>;
  disconnect(): Promise<void>;
  isConnected(): Promise<boolean>;

  // Order Management
  placeOrder(order: OrderRequest): Promise<OrderResponse>;
  cancelOrder(orderId: string): Promise<boolean>;
  cancelAllOrders(symbol?: string): Promise<number>;
  getOrderStatus(orderId: string): Promise<OrderResponse>;
  listOrders(filters?: OrderFilters): Promise<OrderResponse[]>;

  // Account Management
  getAccountBalance(): Promise<AccountBalance>;
  getPositions(): Promise<JsPosition[]>;
  getPosition(symbol: string): Promise<JsPosition | null>;

  // Market Data (Basic)
  getQuote(symbol: string): Promise<Quote>;

  // Configuration
  get config(): BrokerConfig;
  updateConfig(updates: Partial<BrokerConfig>): void;
}
```

### Types

```typescript
interface BrokerConfig {
  brokerType: 'alpaca' | 'interactive_brokers' | 'binance' | 'coinbase';
  apiKey: string;
  apiSecret: string;
  baseUrl?: string;
  paperTrading?: boolean;
  websocketEnabled?: boolean;
  timeout?: number;
}

interface OrderRequest {
  symbol: string;
  side: 'buy' | 'sell';
  orderType: 'market' | 'limit' | 'stop' | 'stop_limit';
  quantity: number;
  limitPrice?: number;
  stopPrice?: number;
  timeInForce: 'day' | 'gtc' | 'ioc' | 'fok' | 'opg' | 'cls';
  extendedHours?: boolean;
  clientOrderId?: string;
}

interface OrderResponse {
  orderId: string;
  clientOrderId?: string;
  symbol: string;
  side: 'buy' | 'sell';
  orderType: string;
  quantity: number;
  filledQuantity: number;
  remainingQuantity: number;
  limitPrice?: number;
  stopPrice?: number;
  averagePrice?: number;
  status: 'pending' | 'open' | 'filled' | 'partially_filled' | 'cancelled' | 'rejected';
  timeInForce: string;
  createdAt: string;
  updatedAt?: string;
  filledAt?: string;
}

interface AccountBalance {
  cash: number;
  equity: number;
  buyingPower: number;
  marginUsed?: number;
  availableForTrading: number;
  currency: string;
}

interface JsPosition {
  symbol: string;
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  marketValue: number;
  costBasis: number;
  unrealizedPl: number;
  unrealizedPlPercent: number;
  side: 'long' | 'short';
}

interface OrderFilters {
  symbol?: string;
  status?: string;
  side?: 'buy' | 'sell';
  limit?: number;
  after?: string;
  until?: string;
}
```

### Utility Functions

```typescript
// List all supported broker types
function listBrokerTypes(): string[];

// Validate broker configuration
function validateBrokerConfig(config: BrokerConfig): boolean;

// Check if broker supports feature
function brokerSupportsFeature(
  brokerType: string,
  feature: 'options' | 'crypto' | 'futures' | 'forex'
): boolean;
```

## Configuration

### Environment Variables

```bash
# Alpaca
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET=your_alpaca_secret

# Interactive Brokers
IB_ACCOUNT=your_account_number
IB_HOST=localhost
IB_PORT=4001

# Binance
BINANCE_API_KEY=your_binance_key
BINANCE_SECRET=your_binance_secret

# Coinbase
COINBASE_API_KEY=your_coinbase_key
COINBASE_SECRET=your_coinbase_secret
COINBASE_PASSPHRASE=your_passphrase
```

### Configuration File

```typescript
// config/brokers.ts
import { BrokerConfig } from '@neural-trader/brokers';

export const brokerConfigs: Record<string, BrokerConfig> = {
  alpaca: {
    brokerType: 'alpaca',
    apiKey: process.env.ALPACA_API_KEY!,
    apiSecret: process.env.ALPACA_SECRET!,
    baseUrl: 'https://paper-api.alpaca.markets',
    paperTrading: true,
    websocketEnabled: true,
    timeout: 30000
  },
  binance: {
    brokerType: 'binance',
    apiKey: process.env.BINANCE_API_KEY!,
    apiSecret: process.env.BINANCE_SECRET!,
    baseUrl: 'https://testnet.binance.vision',
    paperTrading: true,
    timeout: 10000
  }
};
```

## Error Handling

### Common Errors

```typescript
import { BrokerClient } from '@neural-trader/brokers';

const client = new BrokerClient({ /* config */ });

try {
  await client.connect();
  const order = await client.placeOrder({ /* order */ });
} catch (error) {
  if (error.message.includes('Invalid API key')) {
    console.error('Authentication failed - check credentials');
  } else if (error.message.includes('Insufficient funds')) {
    console.error('Not enough buying power for order');
  } else if (error.message.includes('Market closed')) {
    console.error('Market is currently closed');
  } else if (error.message.includes('Rate limit')) {
    console.error('API rate limit exceeded - retry later');
  } else if (error.message.includes('Connection timeout')) {
    console.error('Broker connection timeout - check network');
  } else {
    console.error('Unexpected error:', error);
  }
} finally {
  await client.disconnect();
}
```

### Retry Logic

```typescript
async function connectWithRetry(
  client: BrokerClient,
  maxRetries = 3,
  delayMs = 1000
): Promise<boolean> {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const connected = await client.connect();
      if (connected) return true;
    } catch (error) {
      console.error(`Connection attempt ${i + 1} failed:`, error.message);
      if (i < maxRetries - 1) {
        await new Promise(resolve => setTimeout(resolve, delayMs * (i + 1)));
      }
    }
  }
  throw new Error(`Failed to connect after ${maxRetries} attempts`);
}
```

## Best Practices

### 1. Always Use Paper Trading First

```typescript
// Test with paper trading before live
const client = new BrokerClient({
  brokerType: 'alpaca',
  apiKey: process.env.ALPACA_API_KEY,
  apiSecret: process.env.ALPACA_SECRET,
  paperTrading: true // Always start here
});
```

### 2. Implement Proper Error Handling

```typescript
// Always wrap broker operations in try-catch
try {
  await client.placeOrder(order);
} catch (error) {
  // Log and handle errors appropriately
  console.error('Order failed:', error);
  // Notify monitoring system
  // Attempt recovery or rollback
}
```

### 3. Monitor Order Execution

```typescript
// Don't assume orders are filled immediately
const order = await client.placeOrder(orderRequest);
const finalStatus = await waitForOrderCompletion(client, order.orderId);
if (finalStatus.status !== 'filled') {
  console.warn('Order not filled:', finalStatus);
}
```

### 4. Use Connection Pooling

```typescript
// Reuse broker connections
let clientInstance: BrokerClient | null = null;

function getClient(): BrokerClient {
  if (!clientInstance) {
    clientInstance = new BrokerClient({ /* config */ });
  }
  return clientInstance;
}

// Cleanup on app shutdown
process.on('SIGTERM', async () => {
  if (clientInstance) {
    await clientInstance.disconnect();
  }
});
```

### 5. Respect Rate Limits

```typescript
// Implement rate limiting for API calls
import pLimit from 'p-limit';

const limit = pLimit(5); // Max 5 concurrent requests

const orders = symbols.map(symbol =>
  limit(() => client.placeOrder({
    symbol,
    side: 'buy',
    orderType: 'market',
    quantity: 100,
    timeInForce: 'day'
  }))
);

await Promise.all(orders);
```

### 6. Secure Credential Management

```typescript
// Never hardcode credentials
// Use environment variables or secure vaults
import { BrokerClient } from '@neural-trader/brokers';

const client = new BrokerClient({
  brokerType: 'alpaca',
  apiKey: process.env.ALPACA_API_KEY!, // From environment
  apiSecret: process.env.ALPACA_SECRET!,
  paperTrading: process.env.NODE_ENV !== 'production'
});
```

## License

Dual-licensed under MIT OR Apache-2.0.

See [LICENSE-MIT](../../LICENSE-MIT) and [LICENSE-APACHE](../../LICENSE-APACHE) for details.
