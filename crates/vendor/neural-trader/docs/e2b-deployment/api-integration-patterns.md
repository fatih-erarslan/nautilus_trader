# E2B API Integration Patterns

## Overview

This document outlines best practices and patterns for integrating Alpaca Trading API with E2B sandbox-deployed strategies.

## Core Integration Components

### 1. Alpaca Client Initialization

```javascript
const Alpaca = require('@alpacahq/alpaca-trade-api');

// Standard initialization pattern
const alpaca = new Alpaca({
  keyId: process.env.ALPACA_API_KEY,
  secretKey: process.env.ALPACA_SECRET_KEY,
  paper: true, // Always use paper trading initially
  baseUrl: process.env.ALPACA_BASE_URL || 'https://paper-api.alpaca.markets'
});

// With error handling
function initializeAlpaca() {
  try {
    if (!process.env.ALPACA_API_KEY || !process.env.ALPACA_SECRET_KEY) {
      throw new Error('Missing required Alpaca credentials');
    }

    return new Alpaca({
      keyId: process.env.ALPACA_API_KEY,
      secretKey: process.env.ALPACA_SECRET_KEY,
      paper: true,
      baseUrl: process.env.ALPACA_BASE_URL || 'https://paper-api.alpaca.markets'
    });
  } catch (error) {
    console.error('Failed to initialize Alpaca client:', error);
    process.exit(1);
  }
}
```

### 2. Market Data Fetching

#### Pattern: Historical Bars with Streaming

```javascript
/**
 * Fetch historical bars with automatic pagination
 */
async function getHistoricalBars(symbol, options = {}) {
  const {
    timeframe = '5Min',
    limit = 1000,
    startDate = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    endDate = new Date()
  } = options;

  try {
    const bars = alpaca.getBarsV2(symbol, {
      start: startDate.toISOString(),
      end: endDate.toISOString(),
      timeframe: timeframe,
      limit: limit
    });

    const barArray = [];
    for await (let bar of bars) {
      barArray.push({
        timestamp: bar.Timestamp,
        open: bar.OpenPrice,
        high: bar.HighPrice,
        low: bar.LowPrice,
        close: bar.ClosePrice,
        volume: bar.Volume
      });
    }

    return barArray;

  } catch (error) {
    if (error.message.includes('rate limit')) {
      // Exponential backoff
      await new Promise(resolve => setTimeout(resolve, 5000));
      return getHistoricalBars(symbol, options);
    }
    throw error;
  }
}
```

#### Pattern: Real-time Quotes

```javascript
/**
 * Get latest quote with retry logic
 */
async function getLatestQuote(symbol, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const quote = await alpaca.getLatestTrade(symbol);
      return {
        price: quote.Price,
        timestamp: quote.Timestamp,
        size: quote.Size
      };
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
    }
  }
}
```

### 3. Order Management

#### Pattern: Safe Order Placement

```javascript
/**
 * Place order with comprehensive validation
 */
async function placeOrder(params) {
  const {
    symbol,
    qty,
    side,
    type = 'market',
    timeInForce = 'day',
    limitPrice = null,
    stopPrice = null
  } = params;

  try {
    // Pre-trade validation
    const clock = await alpaca.getClock();
    if (!clock.is_open) {
      throw new Error('Market is closed');
    }

    // Check buying power
    const account = await alpaca.getAccount();
    const estimatedCost = qty * (limitPrice || await getLatestQuote(symbol).price);

    if (side === 'buy' && estimatedCost > parseFloat(account.buying_power)) {
      throw new Error('Insufficient buying power');
    }

    // Check existing position
    const currentPosition = await getPosition(symbol);
    if (side === 'sell' && qty > Math.abs(currentPosition)) {
      throw new Error('Cannot sell more than current position');
    }

    // Place order
    const order = await alpaca.createOrder({
      symbol,
      qty,
      side,
      type,
      time_in_force: timeInForce,
      limit_price: limitPrice,
      stop_price: stopPrice
    });

    // Log trade
    console.log(JSON.stringify({
      level: 'TRADE',
      action: 'ORDER_PLACED',
      orderId: order.id,
      symbol,
      qty,
      side,
      type,
      timestamp: new Date().toISOString()
    }));

    return order;

  } catch (error) {
    console.error(JSON.stringify({
      level: 'ERROR',
      action: 'ORDER_FAILED',
      symbol,
      error: error.message,
      timestamp: new Date().toISOString()
    }));
    throw error;
  }
}
```

#### Pattern: Order Status Monitoring

```javascript
/**
 * Monitor order until filled or timeout
 */
async function waitForOrderFill(orderId, timeoutMs = 60000) {
  const startTime = Date.now();

  while (Date.now() - startTime < timeoutMs) {
    try {
      const order = await alpaca.getOrder(orderId);

      if (order.status === 'filled') {
        return {
          status: 'filled',
          filledQty: order.filled_qty,
          filledPrice: order.filled_avg_price,
          filledAt: order.filled_at
        };
      }

      if (['canceled', 'expired', 'rejected'].includes(order.status)) {
        return {
          status: order.status,
          reason: order.cancel_requested_at || 'Unknown'
        };
      }

      // Wait before checking again
      await new Promise(resolve => setTimeout(resolve, 1000));

    } catch (error) {
      console.error('Error checking order status:', error);
      throw error;
    }
  }

  throw new Error('Order monitoring timeout');
}
```

### 4. Position Management

#### Pattern: Safe Position Retrieval

```javascript
/**
 * Get current position with error handling
 */
async function getPosition(symbol) {
  try {
    const position = await alpaca.getPosition(symbol);
    return {
      qty: parseInt(position.qty),
      side: parseInt(position.qty) > 0 ? 'long' : 'short',
      marketValue: parseFloat(position.market_value),
      costBasis: parseFloat(position.cost_basis),
      unrealizedPL: parseFloat(position.unrealized_pl),
      unrealizedPLPct: parseFloat(position.unrealized_plpc),
      currentPrice: parseFloat(position.current_price)
    };
  } catch (error) {
    if (error.message.includes('position does not exist')) {
      return {
        qty: 0,
        side: 'none',
        marketValue: 0,
        costBasis: 0,
        unrealizedPL: 0,
        unrealizedPLPct: 0,
        currentPrice: 0
      };
    }
    throw error;
  }
}

/**
 * Get all positions with filtering
 */
async function getAllPositions(filter = {}) {
  try {
    const positions = await alpaca.getPositions();

    return positions
      .map(p => ({
        symbol: p.symbol,
        qty: parseInt(p.qty),
        side: parseInt(p.qty) > 0 ? 'long' : 'short',
        marketValue: parseFloat(p.market_value),
        unrealizedPL: parseFloat(p.unrealized_pl),
        unrealizedPLPct: parseFloat(p.unrealized_plpc)
      }))
      .filter(p => {
        if (filter.minPL && p.unrealizedPLPct < filter.minPL) return false;
        if (filter.maxPL && p.unrealizedPLPct > filter.maxPL) return false;
        if (filter.side && p.side !== filter.side) return false;
        return true;
      });
  } catch (error) {
    console.error('Error fetching positions:', error);
    return [];
  }
}
```

### 5. Account Monitoring

#### Pattern: Account Health Check

```javascript
/**
 * Comprehensive account health check
 */
async function checkAccountHealth() {
  try {
    const account = await alpaca.getAccount();

    const health = {
      equity: parseFloat(account.equity),
      cash: parseFloat(account.cash),
      buyingPower: parseFloat(account.buying_power),
      portfolioValue: parseFloat(account.portfolio_value),
      daytradeCount: parseInt(account.daytrade_count),
      patternDayTrader: account.pattern_day_trader,
      tradingBlocked: account.trading_blocked,
      accountBlocked: account.account_blocked,
      transfersBlocked: account.transfers_blocked
    };

    // Health warnings
    const warnings = [];

    if (health.tradingBlocked) {
      warnings.push('TRADING_BLOCKED');
    }

    if (health.daytradeCount >= 3 && !health.patternDayTrader) {
      warnings.push('APPROACHING_PDT_LIMIT');
    }

    if (health.buyingPower < health.equity * 0.1) {
      warnings.push('LOW_BUYING_POWER');
    }

    return { health, warnings };

  } catch (error) {
    console.error('Error checking account health:', error);
    throw error;
  }
}
```

### 6. Risk Controls

#### Pattern: Pre-Trade Risk Check

```javascript
/**
 * Validate trade against risk limits
 */
async function validateTradeRisk(params) {
  const { symbol, qty, side, estimatedPrice } = params;

  try {
    const account = await alpaca.getAccount();
    const equity = parseFloat(account.equity);
    const position = await getPosition(symbol);

    // Calculate trade value
    const tradeValue = qty * estimatedPrice;
    const positionValue = Math.abs(position.qty * position.currentPrice);
    const newPositionValue = side === 'buy'
      ? positionValue + tradeValue
      : positionValue - tradeValue;

    // Risk checks
    const checks = {
      maxPositionSize: newPositionValue <= equity * 0.20, // Max 20% per position
      maxTradeSize: tradeValue <= equity * 0.10, // Max 10% per trade
      buyingPower: side === 'buy' ? tradeValue <= parseFloat(account.buying_power) : true,
      stopLossDistance: true // Implement stop loss validation
    };

    const passed = Object.values(checks).every(check => check === true);

    if (!passed) {
      const failed = Object.entries(checks)
        .filter(([_, passed]) => !passed)
        .map(([check, _]) => check);

      throw new Error(`Risk check failed: ${failed.join(', ')}`);
    }

    return { passed: true, checks };

  } catch (error) {
    console.error('Risk validation error:', error);
    throw error;
  }
}
```

### 7. Error Handling Patterns

#### Pattern: Comprehensive Error Handler

```javascript
/**
 * Centralized error handling for Alpaca API
 */
function handleAlpacaError(error, context = {}) {
  const errorResponse = {
    timestamp: new Date().toISOString(),
    context,
    error: {
      message: error.message,
      code: error.code,
      statusCode: error.statusCode
    }
  };

  // Rate limit errors
  if (error.message.includes('rate limit')) {
    errorResponse.action = 'RETRY_AFTER_DELAY';
    errorResponse.retryAfter = 5000;
    console.error(JSON.stringify({ level: 'ERROR', ...errorResponse }));
    return errorResponse;
  }

  // Market closed errors
  if (error.message.includes('market is closed')) {
    errorResponse.action = 'WAIT_FOR_MARKET_OPEN';
    console.warn(JSON.stringify({ level: 'WARN', ...errorResponse }));
    return errorResponse;
  }

  // Insufficient funds
  if (error.message.includes('insufficient')) {
    errorResponse.action = 'REDUCE_POSITION_SIZE';
    console.error(JSON.stringify({ level: 'ERROR', ...errorResponse }));
    return errorResponse;
  }

  // Authentication errors
  if (error.statusCode === 401 || error.statusCode === 403) {
    errorResponse.action = 'CHECK_CREDENTIALS';
    console.error(JSON.stringify({ level: 'CRITICAL', ...errorResponse }));
    process.exit(1); // Exit on auth errors
  }

  // Generic errors
  errorResponse.action = 'LOG_AND_CONTINUE';
  console.error(JSON.stringify({ level: 'ERROR', ...errorResponse }));
  return errorResponse;
}
```

### 8. Data Streaming Patterns

#### Pattern: WebSocket Price Streaming

```javascript
/**
 * Stream real-time prices using WebSocket
 */
function streamPrices(symbols, onUpdate) {
  const stream = alpaca.data_stream_v2;

  stream.onConnect(() => {
    console.log('Stream connected');
    stream.subscribeForTrades(symbols);
  });

  stream.onStockTrade((trade) => {
    onUpdate({
      symbol: trade.Symbol,
      price: trade.Price,
      size: trade.Size,
      timestamp: trade.Timestamp
    });
  });

  stream.onError((error) => {
    console.error('Stream error:', error);
  });

  stream.onDisconnect(() => {
    console.log('Stream disconnected');
    // Reconnect logic
    setTimeout(() => stream.connect(), 5000);
  });

  stream.connect();

  return stream;
}
```

## Complete Integration Example

```javascript
/**
 * Complete trading strategy with all patterns
 */
class TradingStrategy {
  constructor() {
    this.alpaca = initializeAlpaca();
    this.positions = new Map();
    this.orders = new Map();
  }

  async initialize() {
    // Check account health
    const { health, warnings } = await checkAccountHealth();

    if (warnings.length > 0) {
      console.warn('Account warnings:', warnings);
    }

    // Check market status
    const clock = await this.alpaca.getClock();
    console.log('Market status:', {
      isOpen: clock.is_open,
      nextOpen: clock.next_open,
      nextClose: clock.next_close
    });

    return { health, marketOpen: clock.is_open };
  }

  async executeStrategy(symbol) {
    try {
      // 1. Get market data
      const bars = await getHistoricalBars(symbol, {
        timeframe: '5Min',
        limit: 100
      });

      // 2. Calculate signals
      const signal = this.calculateSignal(bars);

      if (!signal.shouldTrade) {
        return { traded: false, reason: 'No signal' };
      }

      // 3. Get current position
      const position = await getPosition(symbol);

      // 4. Determine trade parameters
      const tradeParams = {
        symbol,
        qty: signal.qty,
        side: signal.side,
        estimatedPrice: bars[bars.length - 1].close
      };

      // 5. Validate risk
      await validateTradeRisk(tradeParams);

      // 6. Place order
      const order = await placeOrder({
        ...tradeParams,
        type: 'market',
        timeInForce: 'day'
      });

      // 7. Monitor order
      const result = await waitForOrderFill(order.id);

      return {
        traded: true,
        order,
        result
      };

    } catch (error) {
      handleAlpacaError(error, { strategy: this.constructor.name, symbol });
      return { traded: false, error: error.message };
    }
  }

  calculateSignal(bars) {
    // Implement strategy logic
    return {
      shouldTrade: false,
      side: 'buy',
      qty: 10
    };
  }
}
```

## Environment Configuration

```javascript
// config.js
module.exports = {
  alpaca: {
    apiKey: process.env.ALPACA_API_KEY,
    secretKey: process.env.ALPACA_SECRET_KEY,
    baseUrl: process.env.ALPACA_BASE_URL || 'https://paper-api.alpaca.markets',
    paper: process.env.ALPACA_PAPER === 'true' || true
  },
  risk: {
    maxPositionSize: parseFloat(process.env.MAX_POSITION_SIZE) || 0.20,
    maxTradeSize: parseFloat(process.env.MAX_TRADE_SIZE) || 0.10,
    stopLossPercent: parseFloat(process.env.STOP_LOSS_PERCENT) || 0.02
  },
  execution: {
    defaultTimeInForce: process.env.TIME_IN_FORCE || 'day',
    maxRetries: parseInt(process.env.MAX_RETRIES) || 3,
    retryDelayMs: parseInt(process.env.RETRY_DELAY_MS) || 1000
  }
};
```

## Best Practices

1. **Always use paper trading initially**: Set `paper: true`
2. **Implement exponential backoff**: Handle rate limits gracefully
3. **Validate before trading**: Check account health and risk limits
4. **Use structured logging**: JSON format for easy parsing
5. **Monitor order status**: Don't assume orders are filled
6. **Handle market hours**: Check if market is open before trading
7. **Implement circuit breakers**: Auto-stop on excessive losses
8. **Cache market data**: Reduce API calls where possible
9. **Use environment variables**: Never hardcode credentials
10. **Test in E2B sandbox**: Isolate strategies for safety
