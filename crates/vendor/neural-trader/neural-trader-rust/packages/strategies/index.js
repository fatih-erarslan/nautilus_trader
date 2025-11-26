// @neural-trader/strategies - Trading strategies package
// Re-exports strategy-related functionality from the main NAPI bindings

const { loadNativeBinary } = require('./load-binary');
const nativeBindings = loadNativeBinary();

const { StrategyRunner, SubscriptionHandle } = nativeBindings;

module.exports = {
  StrategyRunner,
  SubscriptionHandle
};
