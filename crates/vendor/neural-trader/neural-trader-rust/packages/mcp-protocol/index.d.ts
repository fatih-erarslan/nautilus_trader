/**
 * @neural-trader/mcp-protocol - MCP JSON-RPC 2.0 protocol types
 *
 * Type definitions for Model Context Protocol JSON-RPC 2.0 implementation
 */

export type RequestId = string | number;

/**
 * JSON-RPC 2.0 Request
 */
export interface JsonRpcRequest {
  jsonrpc: '2.0';
  method: string;
  params?: any;
  id?: RequestId;
}

/**
 * JSON-RPC 2.0 Error object
 */
export interface JsonRpcError {
  code: number;
  message: string;
  data?: any;
}

/**
 * JSON-RPC 2.0 Response
 */
export interface JsonRpcResponse {
  jsonrpc: '2.0';
  result?: any;
  error?: JsonRpcError;
  id: RequestId;
}

/**
 * Standard JSON-RPC 2.0 error codes
 */
export const ErrorCode: {
  readonly PARSE_ERROR: -32700;
  readonly INVALID_REQUEST: -32600;
  readonly METHOD_NOT_FOUND: -32601;
  readonly INVALID_PARAMS: -32602;
  readonly INTERNAL_ERROR: -32603;
  readonly SERVER_ERROR_START: -32099;
  readonly SERVER_ERROR_END: -32000;
};

/**
 * Create a JSON-RPC 2.0 request
 */
export function createRequest(
  method: string,
  params?: any,
  id?: RequestId
): JsonRpcRequest;

/**
 * Create a JSON-RPC 2.0 success response
 */
export function createSuccessResponse(
  result: any,
  id: RequestId
): JsonRpcResponse;

/**
 * Create a JSON-RPC 2.0 error response
 */
export function createErrorResponse(
  code: number,
  message: string,
  id: RequestId,
  data?: any
): JsonRpcResponse;
