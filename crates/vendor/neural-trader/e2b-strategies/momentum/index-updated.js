/**
 * Momentum Trading Strategy for E2B Sandbox - Updated with neural-trader
 * Symbols: SPY, QQQ, IWM
 * Strategy: Buy on positive momentum > 0.02, fixed position size
 * Now using @neural-trader/* packages for high-performance Rust implementation
 */

const express = require('express');
const { MomentumStrategy } = require('@neural-trader/strategies');
const { MarketDataProvider } = require('@neural-trader/market-data');
const { AlpacaBroker } = require('@neural-trader/brokers');
const { OrderExecutor } = require('@neural-trader/execution');
const { TechnicalIndicators } = require('@neural-trader/features');

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

// Initialize neural-trader components
const broker = new AlpacaBroker({
  apiKey: config.alpacaKey,
  secretKey: config.alpacaSecret,
  baseUrl: config.alpacaBaseUrl,
  paper: true
});

const marketData = new MarketDataProvider({
  provider: 'alpaca',
  apiKey: config.alpacaKey,
  secretKey: config.alpacaSecret
});

const executor = new OrderExecutor({
  broker,
  executionMode: 'market', // Market orders for momentum strategy
  timeout: 30000
});

const strategy = new MomentumStrategy({
  symbols: config.symbols,
  lookbackPeriods: 20,
  threshold: config.momentumThreshold,
  positionSize: config.positionSize,
  timeframe: config.interval
});

// Logging utility
const logger = {
  info: (msg, data = {}) => console.log(JSON.stringify({ level: 'INFO', msg, ...data, timestamp: new Date().toISOString() })),
  error: (msg, data = {}) => console.error(JSON.stringify({ level: 'ERROR', msg, ...data, timestamp: new Date().toISOString() })),
  trade: (msg, data = {}) => console.log(JSON.stringify({ level: 'TRADE', msg, ...data, timestamp: new Date().toISOString() }))
};

/**
 * Calculate momentum using neural-trader features
 */
async function calculateMomentum(symbol) {
  try {
    const endDate = new Date();
    const startDate = new Date(endDate.getTime() - 2 * 60 * 60 * 1000); // 2 hours ago

    // Fetch bars using neural-trader market data
    const bars = await marketData.getBars({
      symbol,
      start: startDate.toISOString(),
      end: endDate.toISOString(),
      timeframe: config.interval,
      limit: 20
    });

    if (!bars || bars.length < 2) {
      logger.error('Insufficient bars for momentum calculation', { symbol, barCount: bars?.length || 0 });
      return null;
    }

    // Use neural-trader technical indicators for momentum calculation
    const prices = bars.map(b => b.close);
    const momentum = TechnicalIndicators.calculateMomentum(prices, 20);

    logger.info('Momentum calculated', {
      symbol,
      currentPrice: bars[bars.length - 1].close,
      momentum: momentum[momentum.length - 1]
    });

    return momentum[momentum.length - 1];

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
    const position = await broker.getPosition(symbol);
    return position?.quantity || 0;
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
    // Use strategy signal generation
    const signal = await strategy.generateSignal(symbol, marketData);

    if (!signal) {
      logger.info('No trade signal', { symbol });
      return;
    }

    const currentPosition = await getPosition(symbol);

    // Buy signal
    if (signal.action === 'buy' && currentPosition === 0) {
      const order = await executor.executeOrder({
        symbol: symbol,
        quantity: config.positionSize,
        side: 'buy',
        type: 'market',
        timeInForce: 'day'
      });

      logger.trade('BUY order placed', {
        symbol,
        momentum: signal.momentum,
        qty: config.positionSize,
        orderId: order.id
      });
    }

    // Sell signal
    else if (signal.action === 'sell' && currentPosition > 0) {
      const order = await executor.executeOrder({
        symbol: symbol,
        quantity: currentPosition,
        side: 'sell',
        type: 'market',
        timeInForce: 'day'
      });

      logger.trade('SELL order placed', {
        symbol,
        momentum: signal.momentum,
        qty: currentPosition,
        orderId: order.id
      });
    }

    else {
      logger.info('No trade action', { symbol, signal, currentPosition });
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

  try {
    // Check market status using broker
    const clock = await broker.getClock();
    if (!clock.isOpen) {
      logger.info('Market is closed', { nextOpen: clock.nextOpen });
      return;
    }

    // Execute trading logic for all symbols in parallel
    await Promise.all(config.symbols.map(symbol => tradingLogic(symbol)));

    logger.info('Strategy cycle completed');
  } catch (error) {
    logger.error('Strategy execution error', { error: error.message });
  }
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
    provider: 'neural-trader',
    timestamp: new Date().toISOString()
  });
});

app.get('/status', async (req, res) => {
  try {
    const account = await broker.getAccount();
    const positions = await broker.getPositions();

    res.json({
      account: {
        equity: account.equity,
        cash: account.cash,
        buyingPower: account.buyingPower
      },
      positions: positions.map(p => ({
        symbol: p.symbol,
        qty: p.quantity,
        currentPrice: p.currentPrice,
        marketValue: p.marketValue,
        unrealizedPL: p.unrealizedPl
      })),
      strategyConfig: {
        threshold: config.momentumThreshold,
        positionSize: config.positionSize,
        interval: config.interval
      }
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
  logger.info('Momentum strategy server started (neural-trader)', { port: config.port });
});

// Run strategy every 5 minutes
setInterval(runStrategy, 5 * 60 * 1000);

// Initial run
runStrategy().catch(error => {
  logger.error('Initial strategy run failed', { error: error.message });
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('SIGTERM received, shutting down gracefully');
  await broker.disconnect();
  await marketData.disconnect();
  process.exit(0);
});
