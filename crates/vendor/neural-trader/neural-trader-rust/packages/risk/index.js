// @neural-trader/risk - Risk management package
// Re-exports risk-related functionality from the main NAPI bindings

const { loadNativeBinary } = require('./load-binary');
const nativeBindings = loadNativeBinary();

const {
  RiskManager,
  calculateSharpeRatio,
  calculateSortinoRatio,
  calculateMaxLeverage
} = nativeBindings;

module.exports = {
  RiskManager,
  calculateSharpeRatio,
  calculateSortinoRatio,
  calculateMaxLeverage
};
