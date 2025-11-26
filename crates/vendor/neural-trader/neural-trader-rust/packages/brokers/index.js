// @neural-trader/brokers - Broker integrations package
// Re-exports broker functionality from the main NAPI bindings

const { loadNativeBinary } = require('./load-binary');
const nativeBindings = loadNativeBinary();

const { BrokerClient, listBrokerTypes, validateBrokerConfig } = nativeBindings;

module.exports = {
  BrokerClient,
  listBrokerTypes,
  validateBrokerConfig
};
