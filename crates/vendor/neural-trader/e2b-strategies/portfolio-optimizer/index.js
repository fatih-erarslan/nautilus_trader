/**
 * Portfolio Optimization Service for E2B Sandbox
 * Implements Sharpe ratio optimization and risk parity allocation
 */

const Alpaca = require('@alpacahq/alpaca-trade-api');
const express = require('express');

// Configuration
const config = {
  alpacaKey: process.env.ALPACA_API_KEY,
  alpacaSecret: process.env.ALPACA_SECRET_KEY,
  alpacaBaseUrl: process.env.ALPACA_BASE_URL || 'https://paper-api.alpaca.markets',
  symbols: ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT', 'AAPL', 'TSLA'], // Diversified universe
  optimizationMethod: 'sharpe', // 'sharpe' or 'risk_parity'
  rebalanceThreshold: 0.05, // 5% deviation triggers rebalance
  lookbackDays: 60,
  targetReturn: null, // null = maximize Sharpe
  riskFreeRate: 0.02 / 252, // ~2% annual
  port: process.env.PORT || 3004
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
  optimize: (msg, data = {}) => console.log(JSON.stringify({ level: 'OPTIMIZE', msg, ...data, timestamp: new Date().toISOString() })),
  rebalance: (msg, data = {}) => console.log(JSON.stringify({ level: 'REBALANCE', msg, ...data, timestamp: new Date().toISOString() }))
};

// Portfolio state
const portfolioState = {
  targetAllocations: null,
  currentAllocations: null,
  expectedReturn: null,
  expectedVolatility: null,
  sharpeRatio: null,
  lastOptimization: null,
  lastRebalance: null
};

/**
 * Get historical returns for all symbols
 */
async function getHistoricalReturns() {
  try {
    const endDate = new Date();
    const startDate = new Date(endDate.getTime() - config.lookbackDays * 24 * 60 * 60 * 1000);

    const returnsMatrix = {};

    for (const symbol of config.symbols) {
      const bars = await alpaca.getBarsV2(symbol, {
        start: startDate.toISOString(),
        end: endDate.toISOString(),
        timeframe: '1D',
        limit: config.lookbackDays
      });

      const barArray = [];
      for await (let bar of bars) {
        barArray.push(bar.ClosePrice);
      }

      // Calculate daily returns
      const returns = [];
      for (let i = 1; i < barArray.length; i++) {
        returns.push((barArray[i] - barArray[i - 1]) / barArray[i - 1]);
      }

      returnsMatrix[symbol] = returns;
    }

    return returnsMatrix;

  } catch (error) {
    logger.error('Error fetching historical returns', { error: error.message });
    return null;
  }
}

/**
 * Calculate mean returns vector
 */
function calculateMeanReturns(returnsMatrix) {
  const means = {};
  for (const [symbol, returns] of Object.entries(returnsMatrix)) {
    means[symbol] = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
  }
  return means;
}

/**
 * Calculate covariance matrix
 */
function calculateCovarianceMatrix(returnsMatrix) {
  const symbols = Object.keys(returnsMatrix);
  const n = symbols.length;
  const covMatrix = Array(n).fill(0).map(() => Array(n).fill(0));

  const means = calculateMeanReturns(returnsMatrix);
  const numReturns = returnsMatrix[symbols[0]].length;

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let covariance = 0;
      const returns_i = returnsMatrix[symbols[i]];
      const returns_j = returnsMatrix[symbols[j]];

      for (let k = 0; k < numReturns; k++) {
        covariance += (returns_i[k] - means[symbols[i]]) * (returns_j[k] - means[symbols[j]]);
      }

      covMatrix[i][j] = covariance / (numReturns - 1);
    }
  }

  return { matrix: covMatrix, symbols };
}

/**
 * Calculate portfolio statistics given weights
 */
function calculatePortfolioStats(weights, meanReturns, covMatrix) {
  const symbols = Object.keys(meanReturns);

  // Expected return = w^T * μ
  let expectedReturn = 0;
  for (let i = 0; i < symbols.length; i++) {
    expectedReturn += weights[i] * meanReturns[symbols[i]];
  }

  // Portfolio variance = w^T * Σ * w
  let variance = 0;
  for (let i = 0; i < symbols.length; i++) {
    for (let j = 0; j < symbols.length; j++) {
      variance += weights[i] * weights[j] * covMatrix.matrix[i][j];
    }
  }

  const volatility = Math.sqrt(variance);
  const sharpe = (expectedReturn - config.riskFreeRate) / volatility;

  return {
    expectedReturn: expectedReturn * 252, // Annualized
    volatility: volatility * Math.sqrt(252), // Annualized
    sharpe
  };
}

/**
 * Optimize portfolio using simplified gradient descent
 * Maximizes Sharpe ratio
 */
function optimizeSharpe(meanReturns, covMatrix) {
  const symbols = Object.keys(meanReturns);
  const n = symbols.length;

  // Initialize with equal weights
  let weights = Array(n).fill(1 / n);
  let bestSharpe = -Infinity;
  let bestWeights = [...weights];

  // Simple random search optimization
  const iterations = 10000;
  for (let iter = 0; iter < iterations; iter++) {
    // Generate random weights that sum to 1
    const randomWeights = Array(n).fill(0).map(() => Math.random());
    const sum = randomWeights.reduce((a, b) => a + b, 0);
    const normalizedWeights = randomWeights.map(w => w / sum);

    const stats = calculatePortfolioStats(normalizedWeights, meanReturns, covMatrix);

    if (stats.sharpe > bestSharpe && !isNaN(stats.sharpe) && isFinite(stats.sharpe)) {
      bestSharpe = stats.sharpe;
      bestWeights = [...normalizedWeights];
    }
  }

  // Convert weights array to object
  const allocations = {};
  symbols.forEach((symbol, i) => {
    allocations[symbol] = bestWeights[i];
  });

  const stats = calculatePortfolioStats(bestWeights, meanReturns, covMatrix);

  return { allocations, stats };
}

/**
 * Risk parity allocation
 * Equal risk contribution from each asset
 */
function optimizeRiskParity(covMatrix) {
  const n = covMatrix.symbols.length;

  // Initialize with equal weights
  let weights = Array(n).fill(1 / n);

  // Iterative risk parity optimization
  const maxIterations = 100;
  const tolerance = 1e-6;

  for (let iter = 0; iter < maxIterations; iter++) {
    const riskContributions = [];
    let totalRisk = 0;

    // Calculate risk contribution of each asset
    for (let i = 0; i < n; i++) {
      let marginalRisk = 0;
      for (let j = 0; j < n; j++) {
        marginalRisk += weights[j] * covMatrix.matrix[i][j];
      }
      const riskContribution = weights[i] * marginalRisk;
      riskContributions.push(riskContribution);
      totalRisk += riskContribution;
    }

    // Adjust weights to equalize risk contributions
    const targetRisk = totalRisk / n;
    let newWeights = weights.map((w, i) =>
      w * Math.sqrt(targetRisk / (riskContributions[i] + 1e-10))
    );

    // Normalize weights
    const sum = newWeights.reduce((a, b) => a + b, 0);
    newWeights = newWeights.map(w => w / sum);

    // Check convergence
    const maxChange = Math.max(...newWeights.map((w, i) => Math.abs(w - weights[i])));
    if (maxChange < tolerance) break;

    weights = newWeights;
  }

  // Convert to allocations object
  const allocations = {};
  covMatrix.symbols.forEach((symbol, i) => {
    allocations[symbol] = weights[i];
  });

  return { allocations };
}

/**
 * Run portfolio optimization
 */
async function optimizePortfolio() {
  try {
    logger.optimize('Starting portfolio optimization', { method: config.optimizationMethod });

    const returnsMatrix = await getHistoricalReturns();
    if (!returnsMatrix) return null;

    const meanReturns = calculateMeanReturns(returnsMatrix);
    const covMatrix = calculateCovarianceMatrix(returnsMatrix);

    let result;
    if (config.optimizationMethod === 'sharpe') {
      result = optimizeSharpe(meanReturns, covMatrix);
    } else {
      result = optimizeRiskParity(covMatrix);
      // Calculate stats for risk parity
      const weights = config.symbols.map(s => result.allocations[s]);
      result.stats = calculatePortfolioStats(weights, meanReturns, covMatrix);
    }

    portfolioState.targetAllocations = result.allocations;
    portfolioState.expectedReturn = result.stats.expectedReturn;
    portfolioState.expectedVolatility = result.stats.volatility;
    portfolioState.sharpeRatio = result.stats.sharpe;
    portfolioState.lastOptimization = new Date().toISOString();

    logger.optimize('Optimization complete', {
      expectedReturn: (result.stats.expectedReturn * 100).toFixed(2) + '%',
      volatility: (result.stats.volatility * 100).toFixed(2) + '%',
      sharpe: result.stats.sharpe.toFixed(2),
      allocations: Object.entries(result.allocations)
        .map(([s, w]) => `${s}: ${(w * 100).toFixed(1)}%`)
        .join(', ')
    });

    return result;

  } catch (error) {
    logger.error('Optimization error', { error: error.message, stack: error.stack });
    return null;
  }
}

/**
 * Calculate current allocations
 */
async function getCurrentAllocations() {
  try {
    const account = await alpaca.getAccount();
    const positions = await alpaca.getPositions();

    const totalValue = parseFloat(account.equity);
    const allocations = {};

    // Initialize all symbols with 0
    config.symbols.forEach(symbol => {
      allocations[symbol] = 0;
    });

    // Calculate current allocations
    positions.forEach(position => {
      const symbol = position.symbol;
      if (config.symbols.includes(symbol)) {
        const marketValue = parseFloat(position.market_value);
        allocations[symbol] = marketValue / totalValue;
      }
    });

    portfolioState.currentAllocations = allocations;
    return allocations;

  } catch (error) {
    logger.error('Error calculating current allocations', { error: error.message });
    return null;
  }
}

/**
 * Check if rebalancing is needed
 */
function needsRebalance(current, target) {
  for (const symbol of config.symbols) {
    const deviation = Math.abs((current[symbol] || 0) - (target[symbol] || 0));
    if (deviation > config.rebalanceThreshold) {
      return true;
    }
  }
  return false;
}

/**
 * Execute rebalancing trades
 */
async function rebalancePortfolio() {
  try {
    if (!portfolioState.targetAllocations) {
      logger.error('No target allocations set');
      return null;
    }

    const currentAllocations = await getCurrentAllocations();
    if (!currentAllocations) return null;

    if (!needsRebalance(currentAllocations, portfolioState.targetAllocations)) {
      logger.info('Rebalancing not needed', { threshold: config.rebalanceThreshold });
      return null;
    }

    logger.rebalance('Starting rebalancing');

    const account = await alpaca.getAccount();
    const totalValue = parseFloat(account.equity);
    const trades = [];

    for (const symbol of config.symbols) {
      const currentWeight = currentAllocations[symbol] || 0;
      const targetWeight = portfolioState.targetAllocations[symbol] || 0;
      const deviation = targetWeight - currentWeight;

      if (Math.abs(deviation) > 0.01) { // 1% minimum trade size
        const targetValue = totalValue * targetWeight;
        const currentValue = totalValue * currentWeight;
        const tradeValue = targetValue - currentValue;

        // Get current price
        const quote = await alpaca.getLatestTrade(symbol);
        const price = quote.Price;
        const qty = Math.floor(Math.abs(tradeValue) / price);

        if (qty > 0) {
          const side = tradeValue > 0 ? 'buy' : 'sell';

          const order = await alpaca.createOrder({
            symbol: symbol,
            qty: qty,
            side: side,
            type: 'market',
            time_in_force: 'day'
          });

          trades.push({
            symbol,
            side,
            qty,
            currentWeight: (currentWeight * 100).toFixed(2) + '%',
            targetWeight: (targetWeight * 100).toFixed(2) + '%',
            orderId: order.id
          });

          logger.rebalance('Rebalance trade executed', {
            symbol,
            side,
            qty,
            currentWeight: (currentWeight * 100).toFixed(2) + '%',
            targetWeight: (targetWeight * 100).toFixed(2) + '%'
          });
        }
      }
    }

    portfolioState.lastRebalance = new Date().toISOString();

    logger.rebalance('Rebalancing complete', { tradeCount: trades.length });
    return trades;

  } catch (error) {
    logger.error('Rebalancing error', { error: error.message, stack: error.stack });
    return null;
  }
}

/**
 * Main optimization and rebalancing loop
 */
async function runOptimizer() {
  logger.info('Running portfolio optimizer');

  // Optimize portfolio
  await optimizePortfolio();

  // Check if rebalancing is needed
  await rebalancePortfolio();

  logger.info('Optimizer cycle completed');
}

/**
 * Express server
 */
const app = express();
app.use(express.json());

app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'portfolio-optimizer',
    timestamp: new Date().toISOString()
  });
});

app.get('/status', (req, res) => {
  res.json({
    ...portfolioState,
    targetAllocations: portfolioState.targetAllocations ?
      Object.entries(portfolioState.targetAllocations)
        .map(([s, w]) => ({ symbol: s, weight: (w * 100).toFixed(2) + '%' })) : null,
    currentAllocations: portfolioState.currentAllocations ?
      Object.entries(portfolioState.currentAllocations)
        .map(([s, w]) => ({ symbol: s, weight: (w * 100).toFixed(2) + '%' })) : null,
    expectedReturn: portfolioState.expectedReturn ? (portfolioState.expectedReturn * 100).toFixed(2) + '%' : null,
    expectedVolatility: portfolioState.expectedVolatility ? (portfolioState.expectedVolatility * 100).toFixed(2) + '%' : null,
    sharpeRatio: portfolioState.sharpeRatio?.toFixed(2)
  });
});

app.post('/optimize', async (req, res) => {
  try {
    const result = await optimizePortfolio();
    res.json({ success: true, result });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/rebalance', async (req, res) => {
  try {
    const trades = await rebalancePortfolio();
    res.json({ success: true, trades });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(config.port, () => {
  logger.info('Portfolio optimizer server started', { port: config.port });
});

// Run optimizer daily (or more frequently in production)
setInterval(runOptimizer, 24 * 60 * 60 * 1000);

// Initial run
runOptimizer().catch(error => {
  logger.error('Initial optimizer run failed', { error: error.message });
});

process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down gracefully');
  process.exit(0);
});
