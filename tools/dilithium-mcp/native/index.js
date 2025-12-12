// Load the native Node.js addon
const nativeBinding = require('./dilithium-native.darwin-x64.node');

// Re-export all functions
module.exports = nativeBinding;
