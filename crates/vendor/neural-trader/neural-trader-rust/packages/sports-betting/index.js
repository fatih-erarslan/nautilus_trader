// @neural-trader/sports-betting - Sports betting package
// Currently using risk management for Kelly criterion
// Will be extended with dedicated sports betting crate

const { loadNativeBinary } = require('./load-binary');
const nativeBindings = loadNativeBinary();

const { RiskManager } = nativeBindings;

module.exports = {
  RiskManager
};
