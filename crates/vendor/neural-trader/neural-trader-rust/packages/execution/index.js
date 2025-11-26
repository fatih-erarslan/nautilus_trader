// @neural-trader/execution - Order execution package
// Re-exports execution functionality from the main NAPI bindings

const { loadNativeBinary } = require('./load-binary');
const nativeBindings = loadNativeBinary();

// Note: Currently using main NeuralTrader class for execution
// In future, will have dedicated execution engine
const { NeuralTrader } = nativeBindings;

module.exports = {
  NeuralTrader
};
