// @neural-trader/backtesting - Backtesting engine package
// Re-exports backtesting functionality from the main NAPI bindings

const { loadNativeBinary } = require('./load-binary');

// Load the platform-appropriate binary
const nativeBindings = loadNativeBinary();

// Export backtesting-specific functions
const { BacktestEngine, compareBacktests } = nativeBindings;

module.exports = {
  BacktestEngine,
  compareBacktests
};
