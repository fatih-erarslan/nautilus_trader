// @neural-trader/neural - Neural network models package
// Re-exports neural model functionality from the main NAPI bindings

const { loadNativeBinary } = require('./load-binary');
const nativeBindings = loadNativeBinary();

const { NeuralModel, BatchPredictor, listModelTypes } = nativeBindings;

module.exports = {
  NeuralModel,
  BatchPredictor,
  listModelTypes
};
