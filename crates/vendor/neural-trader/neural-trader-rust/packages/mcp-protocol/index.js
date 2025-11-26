// @neural-trader/mcp-protocol - MCP JSON-RPC 2.0 protocol types
// Re-exports MCP protocol functionality from the Rust implementation

/**
 * MCP Protocol Types
 *
 * This package provides type-safe JSON-RPC 2.0 protocol types for the
 * Model Context Protocol (MCP) as specified by Anthropic.
 *
 * @module @neural-trader/mcp-protocol
 */

// Note: The Rust crate neural-trader-mcp-protocol provides these types
// but doesn't have NAPI bindings as it's a pure types library.
// For now, we export JavaScript-compatible types.

/**
 * JSON-RPC 2.0 Request
 * @typedef {Object} JsonRpcRequest
 * @property {string} jsonrpc - Always "2.0"
 * @property {string} method - Method name
 * @property {*} [params] - Optional parameters
 * @property {string|number} [id] - Optional request ID
 */

/**
 * JSON-RPC 2.0 Response
 * @typedef {Object} JsonRpcResponse
 * @property {string} jsonrpc - Always "2.0"
 * @property {*} [result] - Result if successful
 * @property {JsonRpcError} [error] - Error if failed
 * @property {string|number} id - Request ID
 */

/**
 * JSON-RPC 2.0 Error
 * @typedef {Object} JsonRpcError
 * @property {number} code - Error code
 * @property {string} message - Error message
 * @property {*} [data] - Additional error data
 */

/**
 * Standard JSON-RPC 2.0 error codes
 */
const ErrorCode = {
  PARSE_ERROR: -32700,
  INVALID_REQUEST: -32600,
  METHOD_NOT_FOUND: -32601,
  INVALID_PARAMS: -32602,
  INTERNAL_ERROR: -32603,
  SERVER_ERROR_START: -32099,
  SERVER_ERROR_END: -32000,
};

/**
 * Create a JSON-RPC 2.0 request
 * @param {string} method - Method name
 * @param {*} [params] - Optional parameters
 * @param {string|number} [id] - Optional request ID
 * @returns {JsonRpcRequest}
 */
function createRequest(method, params, id) {
  const request = {
    jsonrpc: '2.0',
    method,
  };

  if (params !== undefined) {
    request.params = params;
  }

  if (id !== undefined) {
    request.id = id;
  }

  return request;
}

/**
 * Create a JSON-RPC 2.0 success response
 * @param {*} result - Result data
 * @param {string|number} id - Request ID
 * @returns {JsonRpcResponse}
 */
function createSuccessResponse(result, id) {
  return {
    jsonrpc: '2.0',
    result,
    id,
  };
}

/**
 * Create a JSON-RPC 2.0 error response
 * @param {number} code - Error code
 * @param {string} message - Error message
 * @param {string|number} id - Request ID
 * @param {*} [data] - Additional error data
 * @returns {JsonRpcResponse}
 */
function createErrorResponse(code, message, id, data) {
  const error = { code, message };

  if (data !== undefined) {
    error.data = data;
  }

  return {
    jsonrpc: '2.0',
    error,
    id,
  };
}

module.exports = {
  ErrorCode,
  createRequest,
  createSuccessResponse,
  createErrorResponse,
};
