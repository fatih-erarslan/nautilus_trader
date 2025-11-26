// @neural-trader/market-data - Market data providers package
// Re-exports market data functionality from the main NAPI bindings

const { loadNativeBinary } = require('./load-binary');
const nativeBindings = loadNativeBinary();

const {
  MarketDataProvider,
  fetchMarketData,
  listDataProviders,
  encodeBarsToBuffer,
  decodeBarsFromBuffer
} = nativeBindings;

module.exports = {
  MarketDataProvider,
  fetchMarketData,
  listDataProviders,
  encodeBarsToBuffer,
  decodeBarsFromBuffer
};
