/**
 * Risk Management Service for E2B Sandbox
 * Monitors portfolio risk metrics and enforces risk limits
 */

const Alpaca = require('@alpacahq/alpaca-trade-api');
const express = require('express');

// Configuration
const config = {
  alpacaKey: process.env.ALPACA_API_KEY,
  alpacaSecret: process.env.ALPACA_SECRET_KEY,
  alpacaBaseUrl: process.env.ALPACA_BASE_URL || 'https://paper-api.alpaca.markets',
  maxDrawdown: 0.10, // 10% maximum drawdown
  stopLossPerTrade: 0.02, // 2% stop loss per trade
  maxPortfolioRisk: 0.15, // 15% maximum portfolio risk
  varConfidence: 0.95, // 95% confidence for VaR
  lookbackDays: 30,
  port: process.env.PORT || 3003
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
  risk: (msg, data = {}) => console.log(JSON.stringify({ level: 'RISK', msg, ...data, timestamp: new Date().toISOString() })),
  alert: (msg, data = {}) => console.log(JSON.stringify({ level: 'ALERT', msg, ...data, timestamp: new Date().toISOString() }))
};

// Risk metrics storage
const riskMetrics = {
  var: null,
  cvar: null,
  maxDrawdown: null,
  currentDrawdown: null,
  portfolioVolatility: null,
  sharpeRatio: null,
  positions: [],
  alerts: [],
  lastUpdate: null
};

/**
 * Calculate portfolio returns from historical equity
 */
async function getPortfolioReturns() {
  try {
    const endDate = new Date();
    const startDate = new Date(endDate.getTime() - config.lookbackDays * 24 * 60 * 60 * 1000);

    const portfolioHistory = await alpaca.getPortfolioHistory({
      period: `${config.lookbackDays}D`,
      timeframe: '1D'
    });

    const equity = portfolioHistory.equity;
    const returns = [];

    for (let i = 1; i < equity.length; i++) {
      if (equity[i - 1] !== 0) {
        returns.push((equity[i] - equity[i - 1]) / equity[i - 1]);
      }
    }

    return returns;

  } catch (error) {
    logger.error('Error calculating returns', { error: error.message });
    return [];
  }
}

/**
 * Calculate Value at Risk (VaR)
 * VaR = mean return - (z-score * std deviation)
 */
function calculateVaR(returns, confidence = config.varConfidence) {
  if (returns.length === 0) return 0;

  const sortedReturns = [...returns].sort((a, b) => a - b);
  const index = Math.floor((1 - confidence) * sortedReturns.length);
  const var95 = -sortedReturns[index];

  logger.risk('VaR calculated', { var95: (var95 * 100).toFixed(2) + '%', confidence });
  return var95;
}

/**
 * Calculate Conditional Value at Risk (CVaR/Expected Shortfall)
 * CVaR = average of losses beyond VaR threshold
 */
function calculateCVaR(returns, confidence = config.varConfidence) {
  if (returns.length === 0) return 0;

  const sortedReturns = [...returns].sort((a, b) => a - b);
  const index = Math.floor((1 - confidence) * sortedReturns.length);
  const tailLosses = sortedReturns.slice(0, index);

  if (tailLosses.length === 0) return 0;

  const cvar = -tailLosses.reduce((sum, ret) => sum + ret, 0) / tailLosses.length;
  logger.risk('CVaR calculated', { cvar: (cvar * 100).toFixed(2) + '%' });
  return cvar;
}

/**
 * Calculate maximum drawdown
 */
function calculateMaxDrawdown(returns) {
  if (returns.length === 0) return 0;

  let peak = 1;
  let maxDrawdown = 0;
  let currentValue = 1;

  for (const ret of returns) {
    currentValue *= (1 + ret);
    peak = Math.max(peak, currentValue);
    const drawdown = (peak - currentValue) / peak;
    maxDrawdown = Math.max(maxDrawdown, drawdown);
  }

  return maxDrawdown;
}

/**
 * Calculate current drawdown from peak
 */
function calculateCurrentDrawdown(returns) {
  if (returns.length === 0) return 0;

  let peak = 1;
  let currentValue = 1;

  for (const ret of returns) {
    currentValue *= (1 + ret);
    peak = Math.max(peak, currentValue);
  }

  return (peak - currentValue) / peak;
}

/**
 * Calculate portfolio volatility (standard deviation of returns)
 */
function calculateVolatility(returns) {
  if (returns.length < 2) return 0;

  const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
  const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / (returns.length - 1);
  return Math.sqrt(variance);
}

/**
 * Calculate Sharpe Ratio
 * Sharpe = (mean return - risk free rate) / volatility
 */
function calculateSharpeRatio(returns, riskFreeRate = 0.02 / 252) { // ~2% annual
  if (returns.length < 2) return 0;

  const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
  const volatility = calculateVolatility(returns);

  if (volatility === 0) return 0;

  const sharpe = (mean - riskFreeRate) / volatility;
  return sharpe * Math.sqrt(252); // Annualized
}

/**
 * Check stop loss for individual positions
 */
async function checkStopLosses() {
  try {
    const positions = await alpaca.getPositions();
    const alerts = [];

    for (const position of positions) {
      const unrealizedPlPct = parseFloat(position.unrealized_plpc);

      if (unrealizedPlPct <= -config.stopLossPerTrade) {
        alerts.push({
          type: 'STOP_LOSS',
          symbol: position.symbol,
          unrealizedPlPct,
          threshold: -config.stopLossPerTrade,
          action: 'CLOSE_POSITION'
        });

        logger.alert('Stop loss triggered', {
          symbol: position.symbol,
          unrealizedPlPct: (unrealizedPlPct * 100).toFixed(2) + '%',
          threshold: (-config.stopLossPerTrade * 100).toFixed(2) + '%'
        });

        // Auto-close position
        await alpaca.closePosition(position.symbol);
        logger.risk('Position closed due to stop loss', { symbol: position.symbol });
      }
    }

    return alerts;

  } catch (error) {
    logger.error('Error checking stop losses', { error: error.message });
    return [];
  }
}

/**
 * Monitor overall risk metrics
 */
async function monitorRisk() {
  try {
    logger.info('Running risk monitoring');

    const returns = await getPortfolioReturns();
    const positions = await alpaca.getPositions();
    const account = await alpaca.getAccount();

    // Calculate all risk metrics
    const var95 = calculateVaR(returns);
    const cvar = calculateCVaR(returns);
    const maxDrawdown = calculateMaxDrawdown(returns);
    const currentDrawdown = calculateCurrentDrawdown(returns);
    const volatility = calculateVolatility(returns);
    const sharpe = calculateSharpeRatio(returns);

    // Check stop losses
    const stopLossAlerts = await checkStopLosses();

    // Check drawdown limit
    const drawdownAlerts = [];
    if (currentDrawdown > config.maxDrawdown) {
      drawdownAlerts.push({
        type: 'MAX_DRAWDOWN',
        currentDrawdown,
        threshold: config.maxDrawdown,
        action: 'REDUCE_RISK'
      });

      logger.alert('Maximum drawdown exceeded', {
        currentDrawdown: (currentDrawdown * 100).toFixed(2) + '%',
        threshold: (config.maxDrawdown * 100).toFixed(2) + '%'
      });
    }

    // Update risk metrics
    riskMetrics.var = var95;
    riskMetrics.cvar = cvar;
    riskMetrics.maxDrawdown = maxDrawdown;
    riskMetrics.currentDrawdown = currentDrawdown;
    riskMetrics.portfolioVolatility = volatility;
    riskMetrics.sharpeRatio = sharpe;
    riskMetrics.positions = positions.map(p => ({
      symbol: p.symbol,
      qty: p.qty,
      marketValue: parseFloat(p.market_value),
      unrealizedPL: parseFloat(p.unrealized_pl),
      unrealizedPLPct: parseFloat(p.unrealized_plpc)
    }));
    riskMetrics.alerts = [...stopLossAlerts, ...drawdownAlerts];
    riskMetrics.lastUpdate = new Date().toISOString();

    logger.risk('Risk metrics updated', {
      var95: (var95 * 100).toFixed(2) + '%',
      cvar: (cvar * 100).toFixed(2) + '%',
      maxDrawdown: (maxDrawdown * 100).toFixed(2) + '%',
      currentDrawdown: (currentDrawdown * 100).toFixed(2) + '%',
      volatility: (volatility * 100).toFixed(2) + '%',
      sharpe: sharpe.toFixed(2),
      alertCount: riskMetrics.alerts.length
    });

    return riskMetrics;

  } catch (error) {
    logger.error('Error monitoring risk', { error: error.message, stack: error.stack });
    return null;
  }
}

/**
 * Express server
 */
const app = express();
app.use(express.json());

app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'risk-manager',
    timestamp: new Date().toISOString()
  });
});

app.get('/metrics', (req, res) => {
  if (!riskMetrics.lastUpdate) {
    return res.status(503).json({ error: 'Metrics not yet available' });
  }

  res.json({
    ...riskMetrics,
    var: (riskMetrics.var * 100).toFixed(2) + '%',
    cvar: (riskMetrics.cvar * 100).toFixed(2) + '%',
    maxDrawdown: (riskMetrics.maxDrawdown * 100).toFixed(2) + '%',
    currentDrawdown: (riskMetrics.currentDrawdown * 100).toFixed(2) + '%',
    portfolioVolatility: (riskMetrics.portfolioVolatility * 100).toFixed(2) + '%',
    sharpeRatio: riskMetrics.sharpeRatio.toFixed(2)
  });
});

app.get('/alerts', (req, res) => {
  res.json({
    alerts: riskMetrics.alerts,
    count: riskMetrics.alerts.length,
    timestamp: riskMetrics.lastUpdate
  });
});

app.post('/monitor', async (req, res) => {
  try {
    const metrics = await monitorRisk();
    res.json({ success: true, metrics });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(config.port, () => {
  logger.info('Risk manager server started', { port: config.port });
});

// Monitor risk every 2 minutes
setInterval(monitorRisk, 2 * 60 * 1000);

// Initial monitoring
monitorRisk().catch(error => {
  logger.error('Initial risk monitoring failed', { error: error.message });
});

process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down gracefully');
  process.exit(0);
});
