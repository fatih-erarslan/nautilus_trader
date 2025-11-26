// @neural-trader/portfolio - Portfolio management package
// Re-exports portfolio functionality from the main NAPI bindings

const { loadNativeBinary } = require('./load-binary');
const nativeBindings = loadNativeBinary();

const { PortfolioManager, PortfolioOptimizer } = nativeBindings;

module.exports = {
  PortfolioManager,
  PortfolioOptimizer
};
