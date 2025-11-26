/**
 * Python Tool Bridge
 * Spawns Python MCP server as child process and communicates via JSON-RPC
 */

const { spawn } = require('child_process');
const path = require('path');
const readline = require('readline');
const { EventEmitter } = require('events');

class PythonBridge extends EventEmitter {
  constructor(options = {}) {
    super();
    this.options = {
      pythonPath: 'python3',
      mcpServerPath: path.join(__dirname, '../../../../../src/mcp/mcp_server_enhanced.py'),
      env: process.env,
      ...options,
    };
    this.process = null;
    this.requestId = 0;
    this.pendingRequests = new Map();
    this.ready = false;
  }

  /**
   * Start Python MCP server
   */
  async start() {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Python bridge timeout'));
      }, 30000);

      this.process = spawn(this.options.pythonPath, [this.options.mcpServerPath], {
        env: this.options.env,
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      // Handle stdout (JSON-RPC responses)
      const rl = readline.createInterface({
        input: this.process.stdout,
        crlfDelay: Infinity,
      });

      rl.on('line', (line) => {
        try {
          if (line.trim().startsWith('{')) {
            const message = JSON.parse(line);
            this.handleResponse(message);
          }
        } catch (error) {
          console.error('Python bridge parse error:', error.message);
        }
      });

      // Handle stderr (logging)
      this.process.stderr.on('data', (data) => {
        const lines = data.toString().split('\n').filter(l => l.trim());
        for (const line of lines) {
          console.error('[Python]', line);

          // Check for ready signal
          if (line.includes('Waiting for requests')) {
            clearTimeout(timeout);
            this.ready = true;
            this.emit('ready');
            resolve();
          }
        }
      });

      // Handle process errors
      this.process.on('error', (error) => {
        clearTimeout(timeout);
        reject(error);
      });

      // Handle process exit
      this.process.on('exit', (code) => {
        this.ready = false;
        this.emit('exit', code);
      });
    });
  }

  /**
   * Stop Python MCP server
   */
  async stop() {
    if (this.process) {
      this.process.kill('SIGTERM');
      this.process = null;
      this.ready = false;
    }
  }

  /**
   * Call a Python tool via JSON-RPC
   */
  async call(method, params = {}) {
    if (!this.ready) {
      throw new Error('Python bridge not ready');
    }

    return new Promise((resolve, reject) => {
      const id = ++this.requestId;
      const request = {
        jsonrpc: '2.0',
        method,
        params,
        id,
      };

      // Store pending request
      this.pendingRequests.set(id, { resolve, reject });

      // Set timeout for request
      const timeout = setTimeout(() => {
        this.pendingRequests.delete(id);
        reject(new Error(`Python tool timeout: ${method}`));
      }, 60000);

      // Store timeout with request
      this.pendingRequests.get(id).timeout = timeout;

      // Send request to Python process
      this.process.stdin.write(JSON.stringify(request) + '\n');
    });
  }

  /**
   * Handle JSON-RPC response from Python
   */
  handleResponse(message) {
    const { id, result, error } = message;

    if (!this.pendingRequests.has(id)) {
      return;
    }

    const pending = this.pendingRequests.get(id);
    clearTimeout(pending.timeout);
    this.pendingRequests.delete(id);

    if (error) {
      pending.reject(new Error(error.message || 'Python tool error'));
    } else {
      pending.resolve(result);
    }
  }

  /**
   * Check if bridge is ready
   */
  isReady() {
    return this.ready;
  }
}

module.exports = { PythonBridge };
