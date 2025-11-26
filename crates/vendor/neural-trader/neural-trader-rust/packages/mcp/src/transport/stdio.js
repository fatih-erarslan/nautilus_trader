/**
 * STDIO Transport Layer for MCP
 * Implements stdio-based JSON-RPC transport
 *
 * Messages are line-delimited JSON on stdin/stdout
 * Logging goes to stderr to avoid protocol contamination
 */

const readline = require('readline');
const { EventEmitter } = require('events');

class StdioTransport extends EventEmitter {
  constructor(options = {}) {
    super();
    this.options = {
      input: process.stdin,
      output: process.stdout,
      logger: console.error.bind(console), // Use stderr for logs
      ...options,
    };
    this.rl = null;
    this.connected = false;
  }

  /**
   * Start the stdio transport
   */
  async start() {
    if (this.connected) {
      throw new Error('Transport already started');
    }

    this.rl = readline.createInterface({
      input: this.options.input,
      output: this.options.output,
      terminal: false,
      crlfDelay: Infinity,
    });

    // Handle incoming messages
    this.rl.on('line', (line) => {
      if (line.trim()) {
        this.emit('message', line.trim());
      }
    });

    // Handle close
    this.rl.on('close', () => {
      this.emit('close');
      this.connected = false;
    });

    // Handle errors
    this.rl.on('error', (error) => {
      this.emit('error', error);
    });

    this.connected = true;
    this.emit('connect');

    this.log('STDIO transport started');
  }

  /**
   * Stop the stdio transport
   */
  async stop() {
    if (!this.connected) {
      return;
    }

    if (this.rl) {
      this.rl.close();
      this.rl = null;
    }

    this.connected = false;
    this.log('STDIO transport stopped');
  }

  /**
   * Send a message
   */
  send(message) {
    if (!this.connected) {
      throw new Error('Transport not connected');
    }

    if (typeof message !== 'string') {
      message = JSON.stringify(message);
    }

    // Write to stdout with newline
    this.options.output.write(message + '\n');
  }

  /**
   * Log to stderr (doesn't interfere with protocol)
   */
  log(...args) {
    if (this.options.logger) {
      this.options.logger(...args);
    }
  }

  /**
   * Check if transport is connected
   */
  isConnected() {
    return this.connected;
  }
}

module.exports = { StdioTransport };
