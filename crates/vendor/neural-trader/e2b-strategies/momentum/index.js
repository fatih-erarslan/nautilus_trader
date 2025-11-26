/**
 * Momentum Trading Strategy for E2B Sandbox
 * Symbols: SPY, QQQ, IWM
 * Strategy: Buy on positive momentum > 0.02, fixed position size
 */

const Alpaca = require('@alpacahq/alpaca-trade-api');
const express = require('express');

// Configuration from environment variables
const config = {
  alpacaKey: process.env.ALPACA_API_KEY,
  alpacaSecret: process.env.ALPACA_SECRET_KEY,
  alpacaBaseUrl: process.env.ALPACA_BASE_URL || 'https://paper-api.alpaca.markets',
  symbols: ['SPY', 'QQQ', 'IWM'],
  momentumThreshold: 0.02,
  positionSize: 10,
  interval: '5Min',
  port: process.env.PORT || 3000
};

// Initialize Alpaca client
const alpaca = new Alpaca({
  keyId: config.alpacaKey,
  secretKey: config.alpacaSecret,
  paper: true,
  baseUrl: config.alpacaBaseUrl
});

// Logging utility
const logger = {
  info: (msg, data = {}) => console.log(JSON.stringify({ level: 'INFO', msg, ...data, timestamp: new Date().toISOString() })),
  error: (msg, data = {}) => console.error(JSON.stringify({ level: 'ERROR', msg, ...data, timestamp: new Date().toISOString() })),
  trade: (msg, data = {}) => console.log(JSON.stringify({ level: 'TRADE', msg, ...data, timestamp: new Date().toISOString() }))
};

// Price history cache
const priceHistory = new Map();

/**
 * Calculate momentum from recent price bars
 * Momentum = (current_price - price_n_bars_ago) / price_n_bars_ago
 */
async function calculateMomentum(symbol) {
  try {
    const endDate = new Date();
    const startDate = new Date(endDate.getTime() - 2 * 60 * 60 * 1000); // 2 hours ago

    const bars = await alpaca.getBarsV2(symbol, {
      start: startDate.toISOString(),
      end: endDate.toISOString(),
      timeframe: config.interval,
      limit: 20
    });

    const barArray = [];
    for await (let bar of bars) {
      barArray.push(bar);
    }

    if (barArray.length < 2) {
      logger.error('Insufficient bars for momentum calculation', { symbol, barCount: barArray.length });
      return null;
    }

    // Store price history
    priceHistory.set(symbol, barArray);

    const currentPrice = barArray[barArray.length - 1].ClosePrice;
    const previousPrice = barArray[0].ClosePrice;
    const momentum = (currentPrice - previousPrice) / previousPrice;

    logger.info('Momentum calculated', { symbol, currentPrice, previousPrice, momentum });
    return momentum;

  } catch (error) {
    logger.error('Error calculating momentum', { symbol, error: error.message });
    return null;
  }
}

/**
 * Check current position for a symbol
 */
async function getPosition(symbol) {
  try {
    const position = await alpaca.getPosition(symbol);
    return parseInt(position.qty);
  } catch (error) {
    if (error.message.includes('position does not exist')) {
      return 0;
    }
    throw error;
  }
}

/**
 * Execute trading logic for a symbol
 */
async function tradingLogic(symbol) {
  try {
    const momentum = await calculateMomentum(symbol);
    if (momentum === null) return;

    const currentPosition = await getPosition(symbol);

    // Buy signal: positive momentum above threshold and no current position
    if (momentum > config.momentumThreshold && currentPosition === 0) {
      const order = await alpaca.createOrder({
        symbol: symbol,
        qty: config.positionSize,
        side: 'buy',
        type: 'market',
        time_in_force: 'day'
      });

      logger.trade('BUY order placed', {
        symbol,
        momentum,
        qty: config.positionSize,
        orderId: order.id
      });
    }

    // Sell signal: negative momentum and we have a position
    else if (momentum < -config.momentumThreshold && currentPosition > 0) {
      const order = await alpaca.createOrder({
        symbol: symbol,
        qty: currentPosition,
        side: 'sell',
        type: 'market',
        time_in_force: 'day'
      });

      logger.trade('SELL order placed', {
        symbol,
        momentum,
        qty: currentPosition,
        orderId: order.id
      });
    }

    else {
      logger.info('No trade signal', { symbol, momentum, currentPosition });
    }

  } catch (error) {
    logger.error('Trading logic error', { symbol, error: error.message, stack: error.stack });
  }
}

/**
 * Main trading loop
 */
async function runStrategy() {
  logger.info('Momentum strategy started', { symbols: config.symbols, threshold: config.momentumThreshold });

  // Check market status
  const clock = await alpaca.getClock();
  if (!clock.is_open) {
    logger.info('Market is closed', { nextOpen: clock.next_open });
    return;
  }

  // Execute trading logic for all symbols in parallel
  await Promise.all(config.symbols.map(symbol => tradingLogic(symbol)));

  logger.info('Strategy cycle completed');
}

/**
 * Express server for health checks and monitoring
 */
const app = express();
app.use(express.json());

app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    strategy: 'momentum',
    symbols: config.symbols,
    timestamp: new Date().toISOString()
  });
});

app.get('/status', async (req, res) => {
  try {
    const account = await alpaca.getAccount();
    const positions = await alpaca.getPositions();

    res.json({
      account: {
        equity: account.equity,
        cash: account.cash,
        buyingPower: account.buying_power
      },
      positions: positions.map(p => ({
        symbol: p.symbol,
        qty: p.qty,
        currentPrice: p.current_price,
        marketValue: p.market_value,
        unrealizedPL: p.unrealized_pl
      })),
      priceHistory: Array.from(priceHistory.keys()).map(symbol => ({
        symbol,
        barCount: priceHistory.get(symbol).length
      }))
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/execute', async (req, res) => {
  try {
    await runStrategy();
    res.json({ success: true, message: 'Strategy executed' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Start server
app.listen(config.port, () => {
  logger.info('Momentum strategy server started', { port: config.port });
});

// Run strategy every 5 minutes
setInterval(runStrategy, 5 * 60 * 1000);

// Initial run
runStrategy().catch(error => {
  logger.error('Initial strategy run failed', { error: error.message });
});

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down gracefully');
  process.exit(0);
});
