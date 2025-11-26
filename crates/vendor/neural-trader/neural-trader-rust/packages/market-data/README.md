# @neural-trader/market-data

[![npm version](https://img.shields.io/npm/v/@neural-trader/market-data.svg)](https://www.npmjs.com/package/@neural-trader/market-data)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/ruvnet/neural-trader)
[![TypeScript](https://img.shields.io/badge/TypeScript-Ready-blue.svg)](https://www.typescriptlang.org/)

**Enterprise-grade market data providers for Neural Trader** supporting real-time streaming and historical data from Alpaca, Polygon, Yahoo Finance, and more. Built with Rust for maximum throughput and minimal latency.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Providers](#supported-providers)
- [Core Concepts](#core-concepts)
- [Examples](#examples)
- [Integration Patterns](#integration-patterns)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Performance Optimization](#performance-optimization)
- [Best Practices](#best-practices)
- [License](#license)

## Features

- **Multiple Providers**: Alpaca, Polygon, Yahoo Finance, IEX Cloud with unified API
- **Real-Time Streaming**: WebSocket-based live quotes and trades with sub-millisecond latency
- **Historical Data**: OHLCV bars with flexible timeframes (1Min, 5Min, 1Hour, 1Day, etc.)
- **Batch Operations**: Fetch quotes for hundreds of symbols in parallel
- **Data Encoding**: Efficient binary encoding for large datasets (10x compression)
- **Smart Caching**: Automatic caching with TTL for frequently accessed data
- **Rate Limiting**: Built-in rate limit management per provider requirements
- **Rust Performance**: Process millions of data points per second
- **Type Safety**: Full TypeScript definitions for all data structures
- **Crypto & Equities**: Support for both traditional and digital asset data
- **Error Recovery**: Automatic reconnection and data gap filling

## Installation

```bash
# Install with required dependencies
npm install @neural-trader/market-data @neural-trader/core

# Or with yarn
yarn add @neural-trader/market-data @neural-trader/core

# Install type definitions
npm install --save-dev @types/node
```

### Dependencies

This package requires:
- `@neural-trader/core` - Core functionality and types

## Quick Start

### Basic Usage

```typescript
import { MarketDataProvider, listDataProviders } from '@neural-trader/market-data';

// List available providers
console.log('Available providers:', listDataProviders());
// Output: ['alpaca', 'polygon', 'yahoo', 'iex']

// Create provider for Alpaca
const provider = new MarketDataProvider({
  provider: 'alpaca',
  apiKey: process.env.ALPACA_API_KEY,
  apiSecret: process.env.ALPACA_SECRET,
  websocketEnabled: true
});

await provider.connect();
console.log('Connected to market data provider');

// Fetch historical bars (OHLCV data)
const bars = await provider.fetchBars(
  'AAPL',
  '2024-01-01',
  '2024-12-31',
  '1Day'
);

console.log(`Fetched ${bars.length} daily bars for AAPL`);
console.log(`First bar:`, bars[0]);
// Output: { symbol: 'AAPL', timestamp: '2024-01-01T00:00:00Z',
//           open: '185.00', high: '187.50', low: '184.00',
//           close: '186.50', volume: '45234000' }

// Get real-time quote
const quote = await provider.getQuote('AAPL');
console.log(`AAPL Quote:`);
console.log(`  Bid: $${quote.bid} x ${quote.bidSize}`);
console.log(`  Ask: $${quote.ask} x ${quote.askSize}`);
console.log(`  Last: $${quote.last}`);

// Subscribe to real-time quotes
const subscription = provider.subscribeQuotes(['AAPL', 'MSFT', 'GOOGL'], (quote) => {
  console.log(`${quote.symbol}: $${quote.last} (${quote.timestamp})`);
});

// Cleanup after 60 seconds
setTimeout(async () => {
  subscription.unsubscribe();
  await provider.disconnect();
}, 60000);
```

## Supported Providers

### Alpaca
- **Data Coverage**: US Equities, ETFs, Crypto
- **Real-Time**: Yes (WebSocket)
- **Historical**: Yes (1Min to 1Day bars)
- **Free Tier**: Yes (delayed quotes)
- **API Base**: `https://data.alpaca.markets`

### Polygon
- **Data Coverage**: US Equities, Options, Crypto, Forex
- **Real-Time**: Yes (WebSocket)
- **Historical**: Yes (tick-level to daily)
- **Free Tier**: Limited (15 min delay)
- **API Base**: `https://api.polygon.io`

### Yahoo Finance
- **Data Coverage**: Global Equities, ETFs, Indices
- **Real-Time**: Delayed (15-20 minutes)
- **Historical**: Yes (daily bars)
- **Free Tier**: Yes
- **API Base**: `https://query1.finance.yahoo.com`

### IEX Cloud
- **Data Coverage**: US Equities
- **Real-Time**: Yes (WebSocket)
- **Historical**: Yes (intraday and daily)
- **Free Tier**: Limited
- **API Base**: `https://cloud.iexapis.com`

## Core Concepts

### Timeframes

Supported timeframes for historical data:

```typescript
// Intraday timeframes
'1Min'   // 1-minute bars
'5Min'   // 5-minute bars
'15Min'  // 15-minute bars
'30Min'  // 30-minute bars
'1Hour'  // 1-hour bars

// Daily and longer
'1Day'   // Daily bars
'1Week'  // Weekly bars
'1Month' // Monthly bars
```

### Data Types

```typescript
// Bar (OHLCV) data
interface Bar {
  symbol: string;
  timestamp: string;  // ISO 8601 format
  open: string;       // Decimal string
  high: string;
  low: string;
  close: string;
  volume: string;
}

// Real-time quote
interface Quote {
  symbol: string;
  timestamp: string;
  bid: number;
  ask: number;
  bidSize: number;
  askSize: number;
  last: number;
  lastSize: number;
  volume: number;
}

// Trade data
interface Trade {
  symbol: string;
  timestamp: string;
  price: number;
  size: number;
  conditions: string[];
}
```

## Examples

### Fetching Historical Data

```typescript
import { MarketDataProvider } from '@neural-trader/market-data';

const provider = new MarketDataProvider({
  provider: 'alpaca',
  apiKey: process.env.ALPACA_API_KEY,
  apiSecret: process.env.ALPACA_SECRET
});

await provider.connect();

// Fetch daily bars for multiple years
const dailyBars = await provider.fetchBars(
  'AAPL',
  '2020-01-01',
  '2024-12-31',
  '1Day'
);

console.log(`Total bars: ${dailyBars.length}`);
console.log(`Date range: ${dailyBars[0].timestamp} to ${dailyBars[dailyBars.length - 1].timestamp}`);

// Fetch intraday 5-minute bars
const intradayBars = await provider.fetchBars(
  'MSFT',
  '2024-11-01',
  '2024-11-13',
  '5Min'
);

console.log(`5-minute bars: ${intradayBars.length}`);

// Process bars for analysis
dailyBars.forEach(bar => {
  const close = parseFloat(bar.close);
  const open = parseFloat(bar.open);
  const change = ((close - open) / open) * 100;
  console.log(`${bar.timestamp}: ${change.toFixed(2)}%`);
});

await provider.disconnect();
```

### Real-Time Streaming

```typescript
import { MarketDataProvider } from '@neural-trader/market-data';

const provider = new MarketDataProvider({
  provider: 'polygon',
  apiKey: process.env.POLYGON_API_KEY,
  websocketEnabled: true
});

await provider.connect();

// Create price tracker
const priceTracker = new Map<string, number>();

// Subscribe to multiple symbols
const symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'];
const subscription = provider.subscribeQuotes(symbols, (quote) => {
  const prevPrice = priceTracker.get(quote.symbol);
  priceTracker.set(quote.symbol, quote.last);

  if (prevPrice) {
    const change = quote.last - prevPrice;
    const changePercent = (change / prevPrice) * 100;
    const direction = change > 0 ? '↑' : change < 0 ? '↓' : '→';

    console.log(
      `${quote.symbol}: $${quote.last.toFixed(2)} ${direction} ` +
      `${changePercent.toFixed(2)}% (Vol: ${quote.volume})`
    );
  }
});

// Monitor for 5 minutes
await new Promise(resolve => setTimeout(resolve, 300000));

subscription.unsubscribe();
await provider.disconnect();
```

### Batch Quote Fetching

```typescript
import { MarketDataProvider } from '@neural-trader/market-data';

const provider = new MarketDataProvider({
  provider: 'alpaca',
  apiKey: process.env.ALPACA_API_KEY,
  apiSecret: process.env.ALPACA_SECRET
});

await provider.connect();

// Fetch quotes for entire portfolio
const portfolio = [
  'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
  'NVDA', 'META', 'NFLX', 'AMD', 'INTC'
];

const quotes = await provider.getQuotesBatch(portfolio);

// Calculate portfolio metrics
let totalValue = 0;
const positions = {
  'AAPL': 100,
  'MSFT': 50,
  'GOOGL': 25,
  // ... more positions
};

quotes.forEach(quote => {
  const shares = positions[quote.symbol] || 0;
  const value = quote.last * shares;
  totalValue += value;

  console.log(`${quote.symbol}: ${shares} shares @ $${quote.last} = $${value.toFixed(2)}`);
});

console.log(`\nTotal Portfolio Value: $${totalValue.toFixed(2)}`);

await provider.disconnect();
```

### Multi-Provider Data Aggregation

```typescript
import { MarketDataProvider } from '@neural-trader/market-data';

// Use multiple providers for redundancy and coverage
const providers = [
  new MarketDataProvider({
    provider: 'alpaca',
    apiKey: process.env.ALPACA_API_KEY,
    apiSecret: process.env.ALPACA_SECRET
  }),
  new MarketDataProvider({
    provider: 'polygon',
    apiKey: process.env.POLYGON_API_KEY
  })
];

// Connect to all providers
await Promise.all(providers.map(p => p.connect()));

// Fetch data from multiple sources
async function getQuoteWithFallback(symbol: string): Promise<Quote> {
  for (const provider of providers) {
    try {
      return await provider.getQuote(symbol);
    } catch (error) {
      console.warn(`Provider ${provider.config.provider} failed, trying next...`);
    }
  }
  throw new Error(`All providers failed for symbol: ${symbol}`);
}

// Usage
try {
  const quote = await getQuoteWithFallback('AAPL');
  console.log(`AAPL: $${quote.last}`);
} catch (error) {
  console.error('Failed to fetch quote:', error);
}

// Cleanup
await Promise.all(providers.map(p => p.disconnect()));
```

### Data Encoding for Large Datasets

```typescript
import {
  MarketDataProvider,
  encodeBarsToBuffer,
  decodeBarsFromBuffer
} from '@neural-trader/market-data';
import fs from 'fs/promises';

const provider = new MarketDataProvider({
  provider: 'alpaca',
  apiKey: process.env.ALPACA_API_KEY,
  apiSecret: process.env.ALPACA_SECRET
});

await provider.connect();

// Fetch large dataset
const bars = await provider.fetchBars(
  'SPY',
  '2020-01-01',
  '2024-12-31',
  '1Min'
);

console.log(`Fetched ${bars.length} bars`);

// Encode to binary for efficient storage
const encoded = encodeBarsToBuffer(bars);
console.log(`Original size: ${JSON.stringify(bars).length} bytes`);
console.log(`Encoded size: ${encoded.length} bytes`);
console.log(`Compression ratio: ${(JSON.stringify(bars).length / encoded.length).toFixed(2)}x`);

// Save to file
await fs.writeFile('spy_1min.bin', encoded);
console.log('Saved to spy_1min.bin');

// Later: decode from file
const fileBuffer = await fs.readFile('spy_1min.bin');
const decoded = decodeBarsFromBuffer(fileBuffer);
console.log(`Decoded ${decoded.length} bars`);

await provider.disconnect();
```

## Integration Patterns

### Integration with Strategies

```typescript
import { MarketDataProvider } from '@neural-trader/market-data';
import { Strategy } from '@neural-trader/strategies';
import { calculateRsi, calculateSma } from '@neural-trader/features';

// Create strategy
const strategy = new Strategy({
  name: 'rsi-oversold-strategy',
  symbols: ['AAPL', 'MSFT', 'GOOGL'],
  parameters: {
    rsiPeriod: 14,
    rsiOversold: 30,
    smaPeriod: 50
  }
});

// Create market data provider
const provider = new MarketDataProvider({
  provider: 'alpaca',
  apiKey: process.env.ALPACA_API_KEY,
  apiSecret: process.env.ALPACA_SECRET,
  websocketEnabled: true
});

await provider.connect();

// Feed real-time data to strategy
strategy.on('barClose', async (bar) => {
  console.log(`New bar for ${bar.symbol}`);

  // Fetch historical data for indicators
  const endDate = new Date(bar.timestamp);
  const startDate = new Date(endDate);
  startDate.setDate(startDate.getDate() - 100); // 100 days lookback

  const historicalBars = await provider.fetchBars(
    bar.symbol,
    startDate.toISOString().split('T')[0],
    endDate.toISOString().split('T')[0],
    '1Day'
  );

  // Calculate indicators
  const closes = historicalBars.map(b => parseFloat(b.close));
  const rsi = calculateRsi(closes, 14);
  const sma = calculateSma(closes, 50);

  const currentRsi = rsi[rsi.length - 1];
  const currentSma = sma[sma.length - 1];
  const currentClose = parseFloat(bar.close);

  console.log(`${bar.symbol}: RSI=${currentRsi.toFixed(2)}, Price=$${currentClose}, SMA50=$${currentSma.toFixed(2)}`);

  // Generate signals
  if (currentRsi < 30 && currentClose > currentSma) {
    strategy.emit('signal', {
      symbol: bar.symbol,
      action: 'buy',
      reason: 'RSI oversold + above SMA50',
      quantity: 100
    });
  }
});

// Subscribe to real-time bars
const subscription = provider.subscribeQuotes(strategy.symbols, (quote) => {
  // Convert quote to bar (simplified - would track full bar in production)
  strategy.emit('barClose', {
    symbol: quote.symbol,
    timestamp: quote.timestamp,
    close: quote.last.toString(),
    volume: quote.volume.toString()
  });
});

// Run for 1 hour
await new Promise(resolve => setTimeout(resolve, 3600000));

subscription.unsubscribe();
await provider.disconnect();
```

### Integration with Backtesting

```typescript
import { MarketDataProvider } from '@neural-trader/market-data';
import { BacktestEngine } from '@neural-trader/backtesting';
import { calculateSma, calculateRsi } from '@neural-trader/features';

// Create backtest engine
const backtest = new BacktestEngine({
  initialCapital: 100000,
  commission: 0.001,
  startDate: '2023-01-01',
  endDate: '2023-12-31'
});

// Create market data provider
const provider = new MarketDataProvider({
  provider: 'alpaca',
  apiKey: process.env.ALPACA_API_KEY,
  apiSecret: process.env.ALPACA_SECRET
});

await provider.connect();

// Fetch historical data for backtesting
const symbols = ['AAPL', 'MSFT', 'GOOGL'];
const historicalData = new Map<string, Bar[]>();

for (const symbol of symbols) {
  const bars = await provider.fetchBars(
    symbol,
    '2023-01-01',
    '2023-12-31',
    '1Day'
  );
  historicalData.set(symbol, bars);
  console.log(`Loaded ${bars.length} bars for ${symbol}`);
}

// Add strategy using historical data
backtest.addStrategy(async (context) => {
  const { symbol, date, portfolio } = context;
  const bars = historicalData.get(symbol);

  if (!bars) return;

  // Get bars up to current date
  const currentBars = bars.filter(b => new Date(b.timestamp) <= date);
  if (currentBars.length < 50) return; // Need minimum history

  // Calculate indicators
  const closes = currentBars.map(b => parseFloat(b.close));
  const rsi = calculateRsi(closes, 14);
  const sma50 = calculateSma(closes, 50);

  const currentRsi = rsi[rsi.length - 1];
  const currentSma = sma50[sma50.length - 1];
  const currentPrice = parseFloat(currentBars[currentBars.length - 1].close);

  // Trading logic
  if (currentRsi < 30 && currentPrice > currentSma) {
    // Buy signal
    const position = portfolio.positions.get(symbol);
    if (!position || position.quantity === 0) {
      await context.placeOrder({
        symbol,
        side: 'buy',
        orderType: 'market',
        quantity: 100,
        timeInForce: 'day'
      });
    }
  } else if (currentRsi > 70) {
    // Sell signal
    const position = portfolio.positions.get(symbol);
    if (position && position.quantity > 0) {
      await context.placeOrder({
        symbol,
        side: 'sell',
        orderType: 'market',
        quantity: position.quantity,
        timeInForce: 'day'
      });
    }
  }
});

// Run backtest
const results = await backtest.run(symbols);

console.log('\nBacktest Results:');
console.log(`  Initial Capital: $${backtest.initialCapital.toFixed(2)}`);
console.log(`  Final Value: $${results.finalValue.toFixed(2)}`);
console.log(`  Total Return: ${(results.totalReturn * 100).toFixed(2)}%`);
console.log(`  Sharpe Ratio: ${results.sharpeRatio.toFixed(2)}`);
console.log(`  Max Drawdown: ${(results.maxDrawdown * 100).toFixed(2)}%`);
console.log(`  Total Trades: ${results.totalTrades}`);
console.log(`  Win Rate: ${(results.winRate * 100).toFixed(2)}%`);

await provider.disconnect();
```

## API Reference

### MarketDataProvider

Main class for fetching market data.

```typescript
class MarketDataProvider {
  constructor(config: MarketDataConfig);

  // Connection Management
  connect(): Promise<boolean>;
  disconnect(): Promise<void>;
  isConnected(): Promise<boolean>;

  // Historical Data
  fetchBars(
    symbol: string,
    start: string,
    end: string,
    timeframe: string
  ): Promise<Bar[]>;

  // Real-Time Data
  getQuote(symbol: string): Promise<Quote>;
  getQuotesBatch(symbols: string[]): Promise<Quote[]>;

  // WebSocket Streaming
  subscribeQuotes(
    symbols: string[],
    callback: (quote: Quote) => void
  ): Subscription;

  subscribeTrades(
    symbols: string[],
    callback: (trade: Trade) => void
  ): Subscription;

  // Configuration
  get config(): MarketDataConfig;
  updateConfig(updates: Partial<MarketDataConfig>): void;
}
```

### Types

```typescript
interface MarketDataConfig {
  provider: 'alpaca' | 'polygon' | 'yahoo' | 'iex';
  apiKey: string;
  apiSecret?: string;
  websocketEnabled?: boolean;
  timeout?: number;
  cacheTTL?: number;
}

interface Bar {
  symbol: string;
  timestamp: string;  // ISO 8601
  open: string;       // Decimal string
  high: string;
  low: string;
  close: string;
  volume: string;
}

interface Quote {
  symbol: string;
  timestamp: string;
  bid: number;
  ask: number;
  bidSize: number;
  askSize: number;
  last: number;
  lastSize: number;
  volume: number;
}

interface Trade {
  symbol: string;
  timestamp: string;
  price: number;
  size: number;
  conditions: string[];
  exchange?: string;
}

interface Subscription {
  unsubscribe(): void;
  isActive(): boolean;
}
```

### Utility Functions

```typescript
// List all supported providers
function listDataProviders(): string[];

// Binary encoding for large datasets
function encodeBarsToBuffer(bars: Bar[]): Buffer;
function decodeBarsFromBuffer(buffer: Buffer): Bar[];

// Data validation
function validateBar(bar: Bar): boolean;
function validateQuote(quote: Quote): boolean;

// Timeframe utilities
function parseTimeframe(timeframe: string): { value: number; unit: string };
function isValidTimeframe(timeframe: string): boolean;
```

## Configuration

### Environment Variables

```bash
# Alpaca
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET=your_alpaca_secret

# Polygon
POLYGON_API_KEY=your_polygon_key

# IEX Cloud
IEX_API_KEY=your_iex_key
```

### Configuration File

```typescript
// config/market-data.ts
import { MarketDataConfig } from '@neural-trader/market-data';

export const dataProviderConfigs: Record<string, MarketDataConfig> = {
  alpaca: {
    provider: 'alpaca',
    apiKey: process.env.ALPACA_API_KEY!,
    apiSecret: process.env.ALPACA_SECRET!,
    websocketEnabled: true,
    timeout: 30000,
    cacheTTL: 5000
  },
  polygon: {
    provider: 'polygon',
    apiKey: process.env.POLYGON_API_KEY!,
    websocketEnabled: true,
    timeout: 10000,
    cacheTTL: 1000
  },
  yahoo: {
    provider: 'yahoo',
    apiKey: '', // No key required
    websocketEnabled: false,
    timeout: 15000,
    cacheTTL: 60000
  }
};
```

## Performance Optimization

### Efficient Data Fetching

```typescript
// Use batch operations instead of sequential
const symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'];

// ❌ Slow: Sequential fetching
for (const symbol of symbols) {
  const quote = await provider.getQuote(symbol);
  console.log(quote);
}

// ✅ Fast: Batch fetching
const quotes = await provider.getQuotesBatch(symbols);
quotes.forEach(quote => console.log(quote));
```

### Binary Encoding for Storage

```typescript
import { encodeBarsToBuffer, decodeBarsFromBuffer } from '@neural-trader/market-data';

// Fetch large dataset
const bars = await provider.fetchBars('SPY', '2020-01-01', '2024-12-31', '1Min');

// Encode to binary (10x smaller than JSON)
const encoded = encodeBarsToBuffer(bars);

// Store or transmit encoded data
await fs.writeFile('data.bin', encoded);

// Decode when needed
const decoded = decodeBarsFromBuffer(encoded);
```

### Caching Strategy

```typescript
const provider = new MarketDataProvider({
  provider: 'alpaca',
  apiKey: process.env.ALPACA_API_KEY,
  apiSecret: process.env.ALPACA_SECRET,
  cacheTTL: 5000 // Cache quotes for 5 seconds
});

// First call: fetches from API
const quote1 = await provider.getQuote('AAPL');

// Second call within 5 seconds: returns cached data
const quote2 = await provider.getQuote('AAPL');
```

## Best Practices

### 1. Use Appropriate Timeframes

```typescript
// For backtesting: Use daily bars
const dailyBars = await provider.fetchBars('AAPL', '2023-01-01', '2023-12-31', '1Day');

// For intraday trading: Use minute bars
const minuteBars = await provider.fetchBars('AAPL', '2024-11-13', '2024-11-13', '1Min');
```

### 2. Handle Rate Limits

```typescript
import pLimit from 'p-limit';

const limit = pLimit(5); // Max 5 concurrent requests

const symbols = ['AAPL', 'MSFT', /* ... 100 more symbols */];
const quotes = await Promise.all(
  symbols.map(symbol =>
    limit(() => provider.getQuote(symbol))
  )
);
```

### 3. Implement Error Recovery

```typescript
async function fetchBarsWithRetry(
  provider: MarketDataProvider,
  symbol: string,
  start: string,
  end: string,
  timeframe: string,
  maxRetries = 3
): Promise<Bar[]> {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await provider.fetchBars(symbol, start, end, timeframe);
    } catch (error) {
      console.error(`Attempt ${i + 1} failed:`, error.message);
      if (i === maxRetries - 1) throw error;
      await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
    }
  }
  throw new Error('Should not reach here');
}
```

### 4. Monitor WebSocket Connections

```typescript
const provider = new MarketDataProvider({
  provider: 'alpaca',
  apiKey: process.env.ALPACA_API_KEY,
  apiSecret: process.env.ALPACA_SECRET,
  websocketEnabled: true
});

await provider.connect();

// Monitor connection status
setInterval(async () => {
  const connected = await provider.isConnected();
  if (!connected) {
    console.warn('Connection lost, reconnecting...');
    await provider.connect();
  }
}, 10000);
```

### 5. Clean Up Resources

```typescript
// Always disconnect when done
try {
  await provider.connect();
  const bars = await provider.fetchBars('AAPL', '2024-01-01', '2024-12-31', '1Day');
  // Process bars...
} finally {
  await provider.disconnect();
}

// Or use with automatic cleanup
async function withProvider<T>(
  config: MarketDataConfig,
  callback: (provider: MarketDataProvider) => Promise<T>
): Promise<T> {
  const provider = new MarketDataProvider(config);
  try {
    await provider.connect();
    return await callback(provider);
  } finally {
    await provider.disconnect();
  }
}
```

## License

Dual-licensed under MIT OR Apache-2.0.

See [LICENSE-MIT](../../LICENSE-MIT) and [LICENSE-APACHE](../../LICENSE-APACHE) for details.
