/**
 * Rust NAPI Tool Bridge
 * Loads the @neural-trader/core NAPI module and provides JSON-RPC interface
 */

const { EventEmitter } = require('events');

class RustBridge extends EventEmitter {
  constructor(options = {}) {
    super();
    this.options = {
      stubMode: false,
      ...options,
    };
    this.napi = null;
    this.ready = false;
    this.loadError = null;
  }

  /**
   * Load Rust NAPI module
   */
  async start() {
    try {
      // Stub mode only if explicitly enabled (for testing)
      if (this.options.stubMode) {
        console.error('⚠️  Stub mode enabled (testing only)');
        this.ready = true;
        this.emit('ready');
        return;
      }

      // Detect and load the NAPI module
      const modulePath = this.options.modulePath || this.detectNapiModule();
      this.napi = require(modulePath);
      this.ready = true;
      this.emit('ready');
      console.error('✅ Rust NAPI module loaded successfully');
    } catch (error) {
      this.loadError = error;
      console.error('❌ Failed to load Rust NAPI module:', error.message);
      console.error('');
      console.error('This package requires the Rust NAPI binary to be present.');
      console.error('Please ensure the package was installed correctly.');
      console.error('');
      throw error; // Fail fast - no stub mode fallback
    }
  }

  /**
   * Detect NAPI module path based on platform
   */
  detectNapiModule() {
    const path = require('path');
    const platform = process.platform;
    const arch = process.arch;

    // Map platform/arch to NAPI triple
    const triple = this.getPlatformTriple(platform, arch);

    // Try multiple possible locations
    const possiblePaths = [
      // Published package (native/ directory)
      path.join(__dirname, `../../native/neural-trader.${triple}.node`),
      path.join(__dirname, '../../native/neural-trader.node'),

      // Local development build
      path.join(__dirname, '../../../../crates/napi-bindings/neural-trader.node'),
      path.join(__dirname, `../../../../crates/napi-bindings/neural-trader.${triple}.node`),

      // Package installation
      path.join(__dirname, '../../../neural-trader.node'),
      path.join(__dirname, `../../../neural-trader.${triple}.node`),

      // Relative to packages
      path.join(__dirname, '../../../../neural-trader.node'),
      path.join(__dirname, `../../../../neural-trader.${triple}.node`),
    ];

    const fs = require('fs');
    for (const modulePath of possiblePaths) {
      if (fs.existsSync(modulePath)) {
        console.error(`🦀 Rust NAPI module loaded from: ${modulePath}`);
        return modulePath;
      }
    }

    throw new Error(`❌ Rust NAPI module not found. Tried: ${possiblePaths.join(', ')}`);
  }

  /**
   * Get platform triple for NAPI module
   */
  getPlatformTriple(platform, arch) {
    const triples = {
      'linux-x64': 'linux-x64-gnu',
      'linux-arm64': 'linux-arm64-gnu',
      'darwin-x64': 'darwin-x64',
      'darwin-arm64': 'darwin-arm64',
      'win32-x64': 'win32-x64-msvc',
    };

    const key = `${platform}-${arch}`;
    return triples[key] || 'linux-x64-gnu';
  }

  /**
   * Stop Rust bridge
   */
  async stop() {
    this.ready = false;
    this.napi = null;
  }

  /**
   * Call a Rust tool via NAPI
   * @param {string} method - Tool name/method
   * @param {Object} params - Tool parameters
   * @returns {Promise<any>} Tool result
   */
  async call(method, params = {}) {
    if (!this.ready) {
      throw this.createError('Rust bridge not ready', -32603);
    }

    // Stub mode only if explicitly enabled
    if (this.options.stubMode) {
      return this.handleStubCall(method, params);
    }

    // NAPI module must be loaded
    if (!this.napi) {
      throw this.createError('Rust NAPI module not loaded', -32603);
    }

    try {
      // Convert tool name to camelCase for NAPI function lookup
      // e.g., "get_sports_odds" -> "getSportsOdds", "ping" -> "ping"
      const camelCaseName = method.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());

      // Call the NAPI function directly by name
      if (typeof this.napi[camelCaseName] !== 'function') {
        throw this.createError(
          `NAPI function '${camelCaseName}' not found (tool: ${method})`,
          -32601,
          {
            availableFunctions: Object.keys(this.napi).filter(k => typeof this.napi[k] === 'function').slice(0, 20),
          }
        );
      }

      // Call the NAPI function with params
      const result = await this.napi[camelCaseName](params);
      return result;
    } catch (error) {
      // Re-throw errors - no fallback
      throw this.createError(`Rust NAPI call failed for ${method}: ${error.message}`, -32603, {
        method,
        params,
        originalError: error.message,
      });
    }
  }

  /**
   * Handle stub call when NAPI module is not available
   */
  handleStubCall(method, params) {
    return {
      tool: method,
      status: 'stub',
      message: 'Rust NAPI module not available - stub implementation',
      arguments: params,
      timestamp: new Date().toISOString(),
      loadError: this.loadError ? this.loadError.message : null,
      stubMode: true,
    };
  }

  /**
   * Create JSON-RPC formatted error
   */
  createError(message, code = -32603, data = null) {
    const error = new Error(message);
    error.code = code;
    error.data = data;
    return error;
  }

  /**
   * Check if bridge is ready
   */
  isReady() {
    return this.ready;
  }

  /**
   * Check if NAPI module is loaded (not in stub mode)
   */
  isNapiLoaded() {
    return this.ready && !this.options.stubMode && this.napi !== null;
  }

  /**
   * Get bridge status
   */
  getStatus() {
    return {
      ready: this.ready,
      stubMode: this.options.stubMode,
      napiLoaded: this.isNapiLoaded(),
      loadError: this.loadError ? this.loadError.message : null,
      platform: process.platform,
      arch: process.arch,
    };
  }
}

module.exports = { RustBridge };
