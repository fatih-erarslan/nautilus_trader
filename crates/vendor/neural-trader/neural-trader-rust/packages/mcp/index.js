/**
 * @neural-trader/mcp - MCP 2025-11 Compliant Server
 *
 * Provides a Model Context Protocol server with 99+ trading tools
 * accessible to AI assistants like Claude Desktop.
 *
 * @module @neural-trader/mcp
 */

const { McpServer } = require('./src/server');
const { JsonRpcHandler, JsonRpcRequest, JsonRpcResponse, JsonRpcError, ErrorCode } = require('./src/protocol/jsonrpc');
const { StdioTransport } = require('./src/transport/stdio');
const { ToolRegistry } = require('./src/discovery/registry');
const { RustBridge, PythonBridge } = require('./src/bridge');
const { createAuditLogger } = require('./src/logging/audit');
const { E2bSwarmToolHandler, registerE2bSwarmTools, getE2bSwarmTools } = require('./src/tools/e2b-swarm');

/**
 * MCP Server configuration
 * @typedef {Object} McpServerConfig
 * @property {string} [transport] - Transport type: 'stdio' (default)
 * @property {number} [port] - Port number for future HTTP transport (default: 3000)
 * @property {string} [host] - Host address (default: 'localhost')
 * @property {string} [toolsDir] - Path to tools directory
 * @property {boolean} [enablePythonBridge] - Enable Python tool bridge (default: true)
 * @property {boolean} [enableAuditLog] - Enable audit logging (default: true)
 */

/**
 * Start the MCP server with configuration
 * @param {McpServerConfig} config - Server configuration
 * @returns {Promise<McpServer>}
 */
async function startServer(config = {}) {
  const server = new McpServer(config);
  await server.start();
  return server;
}

// Re-export protocol types for TypeScript/JS consumers
module.exports = {
  // Main server class
  McpServer,
  startServer,

  // Protocol components
  JsonRpcHandler,
  JsonRpcRequest,
  JsonRpcResponse,
  JsonRpcError,
  ErrorCode,

  // Transport
  StdioTransport,

  // Discovery
  ToolRegistry,

  // Bridges
  RustBridge,
  PythonBridge,

  // Logging
  createAuditLogger,

  // E2B Swarm Tools
  E2bSwarmToolHandler,
  registerE2bSwarmTools,
  getE2bSwarmTools,
};
