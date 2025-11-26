/**
 * JSON-RPC 2.0 Protocol Handler
 * Implements the JSON-RPC 2.0 specification for MCP
 *
 * @see https://www.jsonrpc.org/specification
 */

class JsonRpcError extends Error {
  constructor(code, message, data = null) {
    super(message);
    this.code = code;
    this.data = data;
  }
}

/**
 * JSON-RPC 2.0 Error Codes
 */
const ErrorCode = {
  PARSE_ERROR: -32700,
  INVALID_REQUEST: -32600,
  METHOD_NOT_FOUND: -32601,
  INVALID_PARAMS: -32602,
  INTERNAL_ERROR: -32603,
  SERVER_ERROR: -32000,
};

/**
 * JSON-RPC 2.0 Request
 */
class JsonRpcRequest {
  constructor(method, params = null, id = null) {
    this.jsonrpc = '2.0';
    this.method = method;
    if (params !== null && params !== undefined) {
      this.params = params;
    }
    if (id !== null && id !== undefined) {
      this.id = id;
    }
  }

  isNotification() {
    return this.id === undefined;
  }

  validate() {
    if (this.jsonrpc !== '2.0') {
      throw new JsonRpcError(ErrorCode.INVALID_REQUEST, 'Invalid JSON-RPC version');
    }
    if (typeof this.method !== 'string') {
      throw new JsonRpcError(ErrorCode.INVALID_REQUEST, 'Method must be a string');
    }
    if (this.params !== undefined && typeof this.params !== 'object') {
      throw new JsonRpcError(ErrorCode.INVALID_PARAMS, 'Params must be an object or array');
    }
    return true;
  }
}

/**
 * JSON-RPC 2.0 Response
 */
class JsonRpcResponse {
  constructor(id, result = null, error = null) {
    this.jsonrpc = '2.0';
    this.id = id;

    if (error) {
      this.error = {
        code: error.code,
        message: error.message,
      };
      if (error.data) {
        this.error.data = error.data;
      }
    } else {
      this.result = result;
    }
  }

  toJSON() {
    const response = {
      jsonrpc: this.jsonrpc,
      id: this.id,
    };

    if (this.error) {
      response.error = this.error;
    } else {
      response.result = this.result;
    }

    return response;
  }
}

/**
 * JSON-RPC 2.0 Handler
 */
class JsonRpcHandler {
  constructor() {
    this.methods = new Map();
    this.middleware = [];
  }

  /**
   * Register a JSON-RPC method
   */
  register(method, handler) {
    if (typeof method !== 'string') {
      throw new Error('Method name must be a string');
    }
    if (typeof handler !== 'function') {
      throw new Error('Handler must be a function');
    }
    this.methods.set(method, handler);
  }

  /**
   * Add middleware function
   */
  use(middleware) {
    if (typeof middleware !== 'function') {
      throw new Error('Middleware must be a function');
    }
    this.middleware.push(middleware);
  }

  /**
   * Parse JSON-RPC request from string
   */
  parse(message) {
    try {
      const data = JSON.parse(message);

      // Handle batch requests
      if (Array.isArray(data)) {
        return data.map(item => new JsonRpcRequest(item.method, item.params, item.id));
      }

      return new JsonRpcRequest(data.method, data.params, data.id);
    } catch (error) {
      throw new JsonRpcError(ErrorCode.PARSE_ERROR, 'Parse error', error.message);
    }
  }

  /**
   * Handle a JSON-RPC request
   */
  async handle(request) {
    try {
      // Validate request
      request.validate();

      // Run middleware
      for (const mw of this.middleware) {
        await mw(request);
      }

      // Find method handler
      const handler = this.methods.get(request.method);
      if (!handler) {
        throw new JsonRpcError(
          ErrorCode.METHOD_NOT_FOUND,
          `Method not found: ${request.method}`
        );
      }

      // Execute method
      const result = await handler(request.params || {});

      // Return response (or null for notifications)
      if (request.isNotification()) {
        return null;
      }

      return new JsonRpcResponse(request.id, result);
    } catch (error) {
      if (request.isNotification()) {
        // Don't send errors for notifications
        return null;
      }

      if (error instanceof JsonRpcError) {
        return new JsonRpcResponse(request.id, null, error);
      }

      // Wrap unknown errors
      return new JsonRpcResponse(
        request.id,
        null,
        new JsonRpcError(ErrorCode.INTERNAL_ERROR, error.message, error.stack)
      );
    }
  }

  /**
   * Handle batch requests
   */
  async handleBatch(requests) {
    const responses = await Promise.all(
      requests.map(req => this.handle(req))
    );

    // Filter out notification responses (null)
    return responses.filter(res => res !== null);
  }

  /**
   * Process a message (either single or batch)
   */
  async process(message) {
    try {
      const request = this.parse(message);

      if (Array.isArray(request)) {
        const responses = await this.handleBatch(request);
        return responses.length > 0 ? JSON.stringify(responses) : null;
      }

      const response = await this.handle(request);
      return response ? JSON.stringify(response.toJSON()) : null;
    } catch (error) {
      if (error instanceof JsonRpcError) {
        const response = new JsonRpcResponse(null, null, error);
        return JSON.stringify(response.toJSON());
      }
      throw error;
    }
  }
}

module.exports = {
  JsonRpcHandler,
  JsonRpcRequest,
  JsonRpcResponse,
  JsonRpcError,
  ErrorCode,
};
