/**
 * Mean Reversion Trading Strategy for E2B Sandbox
 * Symbols: GLD, SLV, TLT
 * Strategy: Z-score based mean reversion with position limits
 */

const Alpaca = require('@alpacahq/alpaca-trade-api');
const express = require('express');

// Configuration
const config = {
  alpacaKey: process.env.ALPACA_API_KEY,
  alpacaSecret: process.env.ALPACA_SECRET_KEY,
  alpacaBaseUrl: process.env.ALPACA_BASE_URL || 'https://paper-api.alpaca.markets',
  symbols: ['GLD', 'SLV', 'TLT'],
  lookbackPeriods: 20,
  zScoreBuyThreshold: -2.0,
  zScoreSellThreshold: 2.0,
  zScoreExitThreshold: 0.5,
  maxPositionSize: 100,
  port: process.env.PORT || 3002
};

const alpaca = new Alpaca({
  keyId: config.alpacaKey,
  secretKey: config.alpacaSecret,
  paper: true,
  baseUrl: config.alpacaBaseUrl
});

const logger = {
  info: (msg, data = {}) => console.log(JSON.stringify({ level: 'INFO', msg, ...data, timestamp: new Date().toISOString() })),
  error: (msg, data = {}) => console.error(JSON.stringify({ level: 'ERROR', msg, ...data, timestamp: new Date().toISOString() })),
  trade: (msg, data = {}) => console.log(JSON.stringify({ level: 'TRADE', msg, ...data, timestamp: new Date().toISOString() })),
  signal: (msg, data = {}) => console.log(JSON.stringify({ level: 'SIGNAL', msg, ...data, timestamp: new Date().toISOString() }))
};

// Store statistics for each symbol
const statistics = new Map();

/**
 * Calculate Simple Moving Average
 */
function calculateSMA(prices, period) {
  if (prices.length < period) return null;
  const slice = prices.slice(-period);
  return slice.reduce((sum, price) => sum + price, 0) / period;
}

/**
 * Calculate Standard Deviation
 */
function calculateStdDev(prices, period, mean) {
  if (prices.length < period) return null;
  const slice = prices.slice(-period);
  const variance = slice.reduce((sum, price) => sum + Math.pow(price - mean, 2), 0) / period;
  return Math.sqrt(variance);
}

/**
 * Calculate Z-Score
 * Z-Score = (current_price - SMA) / StdDev
 */
function calculateZScore(currentPrice, sma, stdDev) {
  if (!sma || !stdDev || stdDev === 0) return null;
  return (currentPrice - sma) / stdDev;
}

/**
 * Get historical bars and calculate statistics
 */
async function calculateStatistics(symbol) {
  try {
    const endDate = new Date();
    const startDate = new Date(endDate.getTime() - 5 * 24 * 60 * 60 * 1000); // 5 days

    const bars = await alpaca.getBarsV2(symbol, {
      start: startDate.toISOString(),
      end: endDate.toISOString(),
      timeframe: '5Min',
      limit: 500
    });

    const barArray = [];
    for await (let bar of bars) {
      barArray.push(bar);
    }

    if (barArray.length < config.lookbackPeriods) {
      logger.error('Insufficient bars for statistics', { symbol, barCount: barArray.length });
      return null;
    }

    const prices = barArray.map(b => b.ClosePrice);
    const currentPrice = prices[prices.length - 1];
    const sma = calculateSMA(prices, config.lookbackPeriods);
    const stdDev = calculateStdDev(prices, config.lookbackPeriods, sma);
    const zScore = calculateZScore(currentPrice, sma, stdDev);

    const stats = {
      currentPrice,
      sma,
      stdDev,
      zScore,
      priceHistory: prices,
      timestamp: new Date().toISOString()
    };

    statistics.set(symbol, stats);

    logger.signal('Statistics calculated', {
      symbol,
      currentPrice: currentPrice.toFixed(2),
      sma: sma?.toFixed(2),
      stdDev: stdDev?.toFixed(4),
      zScore: zScore?.toFixed(2)
    });

    return stats;

  } catch (error) {
    logger.error('Error calculating statistics', { symbol, error: error.message });
    return null;
  }
}

/**
 * Check current position
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
 * Execute mean reversion trading logic
 */
async function tradingLogic(symbol) {
  try {
    const stats = await calculateStatistics(symbol);
    if (!stats || stats.zScore === null) return;

    const { zScore, currentPrice } = stats;
    const currentPosition = await getPosition(symbol);

    // Entry signals
    if (zScore <= config.zScoreBuyThreshold && currentPosition === 0) {
      // Oversold - Buy signal
      const order = await alpaca.createOrder({
        symbol: symbol,
        qty: config.maxPositionSize,
        side: 'buy',
        type: 'market',
        time_in_force: 'day'
      });

      logger.trade('BUY order placed (oversold)', {
        symbol,
        zScore: zScore.toFixed(2),
        currentPrice,
        qty: config.maxPositionSize,
        orderId: order.id
      });
    }
    else if (zScore >= config.zScoreSellThreshold && currentPosition === 0) {
      // Overbought - Short signal (sell first, buy back later)
      const order = await alpaca.createOrder({
        symbol: symbol,
        qty: config.maxPositionSize,
        side: 'sell',
        type: 'market',
        time_in_force: 'day'
      });

      logger.trade('SHORT order placed (overbought)', {
        symbol,
        zScore: zScore.toFixed(2),
        currentPrice,
        qty: config.maxPositionSize,
        orderId: order.id
      });
    }

    // Exit signals - mean reversion complete
    else if (Math.abs(zScore) <= config.zScoreExitThreshold && currentPosition !== 0) {
      // Close position when price reverts to mean
      const side = currentPosition > 0 ? 'sell' : 'buy';
      const order = await alpaca.createOrder({
        symbol: symbol,
        qty: Math.abs(currentPosition),
        side: side,
        type: 'market',
        time_in_force: 'day'
      });

      logger.trade('EXIT order placed (mean reversion)', {
        symbol,
        zScore: zScore.toFixed(2),
        currentPrice,
        previousPosition: currentPosition,
        qty: Math.abs(currentPosition),
        orderId: order.id
      });
    }

    else {
      logger.info('No trade signal', {
        symbol,
        zScore: zScore.toFixed(2),
        currentPosition,
        buyThreshold: config.zScoreBuyThreshold,
        sellThreshold: config.zScoreSellThreshold
      });
    }

  } catch (error) {
    logger.error('Trading logic error', { symbol, error: error.message, stack: error.stack });
  }
}

/**
 * Main strategy loop
 */
async function runStrategy() {
  logger.info('Mean reversion strategy started', { symbols: config.symbols });

  const clock = await alpaca.getClock();
  if (!clock.is_open) {
    logger.info('Market is closed', { nextOpen: clock.next_open });
    return;
  }

  await Promise.all(config.symbols.map(symbol => tradingLogic(symbol)));

  logger.info('Strategy cycle completed');
}

/**
 * Express server
 */
const app = express();
app.use(express.json());

app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    strategy: 'mean-reversion',
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
      statistics: Array.from(statistics.entries()).map(([symbol, stats]) => ({
        symbol,
        zScore: stats.zScore?.toFixed(2),
        sma: stats.sma?.toFixed(2),
        currentPrice: stats.currentPrice?.toFixed(2)
      }))
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/statistics/:symbol', (req, res) => {
  const { symbol } = req.params;
  const stats = statistics.get(symbol);

  if (!stats) {
    return res.status(404).json({ error: 'Statistics not found for symbol' });
  }

  res.json({
    symbol,
    currentPrice: stats.currentPrice,
    sma: stats.sma,
    stdDev: stats.stdDev,
    zScore: stats.zScore,
    timestamp: stats.timestamp
  });
});

app.post('/execute', async (req, res) => {
  try {
    await runStrategy();
    res.json({ success: true, message: 'Strategy executed' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(config.port, () => {
  logger.info('Mean reversion server started', { port: config.port });
});

// Run strategy every 5 minutes
setInterval(runStrategy, 5 * 60 * 1000);

// Initial run
runStrategy().catch(error => {
  logger.error('Initial strategy run failed', { error: error.message });
});

process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down gracefully');
  process.exit(0);
});
