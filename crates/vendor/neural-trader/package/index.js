/**
 * Neural Trader - NAPI Bindings Entry Point
 * Provides JavaScript/TypeScript interface to Rust implementation
 */

const { platform, arch } = process;

// Platform-specific binary loading
const nativeBinding = (() => {
  const loadErrors = [];

  // Try to load platform-specific native module
  const targets = [
    [`${platform}-${arch}`, `./neural-trader-rust/crates/napi-bindings/neural-trader.${platform}-${arch}.node`],
    ['universal', './neural-trader-rust/crates/napi-bindings/neural-trader.node']
  ];

  for (const [name, path] of targets) {
    try {
      return require(path);
    } catch (error) {
      loadErrors.push(`[${name}]: ${error.message}`);
    }
  }

  throw new Error(
    'Failed to load native binding. Errors:\n' +
    loadErrors.join('\n') +
    '\n\nThis usually means:\n' +
    '1. Native bindings not built for your platform\n' +
    '2. Run: npm run build:napi\n' +
    '3. Or use CLI fallback: npx neural-trader'
  );
})();

/**
 * Market Data Functions
 */
function fetchMarketData(symbol, startDate, endDate, provider = 'alpaca') {
  return nativeBinding.fetchMarketData(symbol, startDate, endDate, provider);
}

function streamMarketData(symbol, callback, provider = 'alpaca') {
  return nativeBinding.streamMarketData(symbol, callback, provider);
}

/**
 * Strategy Functions
 */
function runStrategy(strategyType, config) {
  return nativeBinding.runStrategy(strategyType, config);
}

function backtest(strategyType, config, startDate, endDate) {
  return nativeBinding.backtest(strategyType, config, startDate, endDate);
}

/**
 * Execution Functions
 */
function executeOrder(order) {
  return nativeBinding.executeOrder(order);
}

function cancelOrder(orderId) {
  return nativeBinding.cancelOrder(orderId);
}

function getOrderStatus(orderId) {
  return nativeBinding.getOrderStatus(orderId);
}

/**
 * Portfolio Functions
 */
function getPortfolio() {
  return nativeBinding.getPortfolio();
}

function getPositions() {
  return nativeBinding.getPositions();
}

function calculateMetrics(portfolio) {
  return nativeBinding.calculateMetrics(portfolio);
}

/**
 * Risk Management Functions
 */
function calculateVaR(portfolio, confidence = 0.95, method = 'historical') {
  return nativeBinding.calculateVaR(portfolio, confidence, method);
}

function calculatePositionSize(symbol, accountValue, riskPercent, stopLoss) {
  return nativeBinding.calculatePositionSize(symbol, accountValue, riskPercent, stopLoss);
}

function checkRiskLimits(portfolio, limits) {
  return nativeBinding.checkRiskLimits(portfolio, limits);
}

/**
 * Neural Network Functions
 */
function trainModel(modelType, data, config) {
  return nativeBinding.trainModel(modelType, data, config);
}

function predict(modelId, input) {
  return nativeBinding.predict(modelId, input);
}

function evaluateModel(modelId, testData) {
  return nativeBinding.evaluateModel(modelId, testData);
}

/**
 * CLI Runner (for bin/cli.js)
 */
function runCli(args) {
  return nativeBinding.runCli(args);
}

module.exports = {
  // Market Data
  fetchMarketData,
  streamMarketData,

  // Strategy
  runStrategy,
  backtest,

  // Execution
  executeOrder,
  cancelOrder,
  getOrderStatus,

  // Portfolio
  getPortfolio,
  getPositions,
  calculateMetrics,

  // Risk
  calculateVaR,
  calculatePositionSize,
  checkRiskLimits,

  // Neural
  trainModel,
  predict,
  evaluateModel,

  // CLI
  runCli
};
