/**
 * Bridge exports
 * Provides unified interface for tool execution backends
 */

const { RustBridge } = require('./rust');
const { PythonBridge } = require('./python');

module.exports = {
  RustBridge,
  PythonBridge,
};
