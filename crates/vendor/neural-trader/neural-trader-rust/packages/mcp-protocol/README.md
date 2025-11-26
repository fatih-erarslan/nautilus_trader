# @neural-trader/mcp-protocol

[![CI Status](https://github.com/ruvnet/neural-trader/workflows/Rust%20CI/badge.svg)](https://github.com/ruvnet/neural-trader/actions)
[![codecov](https://codecov.io/gh/ruvnet/neural-trader/branch/main/graph/badge.svg)](https://codecov.io/gh/ruvnet/neural-trader)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](../../LICENSE)
[![npm version](https://badge.fury.io/js/%40neural-trader%2Fmcp-protocol.svg)](https://www.npmjs.com/package/@neural-trader/mcp-protocol)

**Type-safe JSON-RPC 2.0 implementation for Model Context Protocol (MCP) - The foundation for AI-first trading tools**

## Introduction

`@neural-trader/mcp-protocol` provides a complete, type-safe implementation of JSON-RPC 2.0 protocol types specifically designed for the Model Context Protocol (MCP). It serves as the communication layer between AI assistants like Claude and Neural Trader's trading capabilities.

This package is the foundation for [@neural-trader/mcp](https://www.npmjs.com/package/@neural-trader/mcp), providing all protocol-level types, request/response structures, and error handling for bidirectional communication between AI assistants and trading systems.

Built with zero dependencies (except [@neural-trader/core](https://www.npmjs.com/package/@neural-trader/core)), it ensures lightweight, predictable, and standards-compliant protocol communication for AI-driven trading workflows.

## Features

- **JSON-RPC 2.0 Compliant**: Fully implements JSON-RPC 2.0 specification
- **Type-Safe Protocol**: Complete TypeScript definitions for requests, responses, and errors
- **MCP Specification**: Follows Anthropic's Model Context Protocol standards
- **Zero Dependencies**: Only depends on @neural-trader/core for shared types
- **Error Codes**: Standard JSON-RPC 2.0 error codes with type safety
- **Request/Response Helpers**: Utility functions for creating protocol messages
- **Lightweight**: ~10 KB package size with no runtime overhead
- **AI-First Design**: Built specifically for LLM-to-system communication
- **Bidirectional**: Supports both client and server message patterns
- **Validation-Ready**: Type structure supports runtime validation

## Installation

### Via npm (Node.js)

```bash
# Protocol package
npm install @neural-trader/mcp-protocol

# With core types
npm install @neural-trader/mcp-protocol @neural-trader/core

# With MCP server
npm install @neural-trader/mcp-protocol @neural-trader/mcp
```

### Via Cargo (Rust)

```bash
# Add to Cargo.toml
[dependencies]
neural-trader-mcp-protocol = "1.0.0"
neural-trader-core = "1.0.0"
```

**Package Size**: ~10 KB (minimal dependencies)

See [main packages documentation](../README.md) for all available packages.

## Quick Start

**30-second example** showing basic protocol usage:

```javascript
const {
  createRequest,
  createSuccessResponse,
  createErrorResponse,
  ErrorCode
} = require('@neural-trader/mcp-protocol');

// Create a JSON-RPC 2.0 request
const request = createRequest('list_strategies', { category: 'momentum' }, 'req-001');

console.log(request);
// {
//   jsonrpc: '2.0',
//   method: 'list_strategies',
//   params: { category: 'momentum' },
//   id: 'req-001'
// }

// Create a success response
const response = createSuccessResponse(
  { strategies: ['momentum-v1', 'momentum-v2'] },
  'req-001'
);

console.log(response);
// {
//   jsonrpc: '2.0',
//   result: { strategies: ['momentum-v1', 'momentum-v2'] },
//   id: 'req-001'
// }

// Create an error response
const error = createErrorResponse(
  ErrorCode.METHOD_NOT_FOUND,
  'Method not found',
  'req-002'
);

console.log(error);
// {
//   jsonrpc: '2.0',
//   error: { code: -32601, message: 'Method not found' },
//   id: 'req-002'
// }
```

**Expected Output:**
```
{ jsonrpc: '2.0', method: 'list_strategies', params: { category: 'momentum' }, id: 'req-001' }
{ jsonrpc: '2.0', result: { strategies: ['momentum-v1', 'momentum-v2'] }, id: 'req-001' }
{ jsonrpc: '2.0', error: { code: -32601, message: 'Method not found' }, id: 'req-002' }
```

## Core Concepts

### JSON-RPC 2.0 Protocol

JSON-RPC 2.0 is a stateless, light-weight remote procedure call (RPC) protocol.

```javascript
const { createRequest } = require('@neural-trader/mcp-protocol');

// Request with ID (expects response)
const requestWithId = createRequest('method_name', { param: 'value' }, 'unique-id');

// Notification (no ID, no response expected)
const notification = createRequest('notify_event', { data: 'value' });

// Request with complex parameters
const complexRequest = createRequest(
  'execute_trade',
  {
    symbol: 'AAPL',
    side: 'BUY',
    quantity: 100,
    orderType: 'LIMIT',
    limitPrice: 175.50
  },
  'trade-001'
);
```

### Request/Response Correlation

Every request with an ID must receive a response with the same ID.

```javascript
const {
  createRequest,
  createSuccessResponse,
  createErrorResponse,
  ErrorCode
} = require('@neural-trader/mcp-protocol');

// Client creates request
const request = createRequest('get_portfolio', { detailed: true }, 'port-001');

// Server processes and responds
try {
  const portfolioData = { /* ... portfolio data ... */ };
  const response = createSuccessResponse(portfolioData, request.id);
  // Send response back to client
} catch (error) {
  const errorResponse = createErrorResponse(
    ErrorCode.INTERNAL_ERROR,
    error.message,
    request.id
  );
  // Send error response back to client
}
```

### Standard Error Codes

JSON-RPC 2.0 defines standard error codes for common failure scenarios.

```javascript
const { ErrorCode } = require('@neural-trader/mcp-protocol');

// Standard error codes
console.log(ErrorCode.PARSE_ERROR);        // -32700: Invalid JSON
console.log(ErrorCode.INVALID_REQUEST);    // -32600: Invalid request object
console.log(ErrorCode.METHOD_NOT_FOUND);   // -32601: Method does not exist
console.log(ErrorCode.INVALID_PARAMS);     // -32602: Invalid method parameters
console.log(ErrorCode.INTERNAL_ERROR);     // -32603: Internal JSON-RPC error

// Server-defined error range
console.log(ErrorCode.SERVER_ERROR_START); // -32099
console.log(ErrorCode.SERVER_ERROR_END);   // -32000
```

### Key Terminology

- **Request**: A call to a remote method with optional parameters and ID
- **Response**: Result or error returned for a request with matching ID
- **Notification**: A request without an ID that expects no response
- **Error Object**: Structured error with code, message, and optional data
- **Request ID**: Unique identifier for correlating requests with responses

### Architecture Overview

```
┌────────────────────────────────────────────────┐
│     @neural-trader/mcp-protocol (Protocol)      │
│                                                 │
│  ┌──────────────┐        ┌──────────────┐     │
│  │   JSON-RPC   │        │    Error     │     │
│  │   Requests   │        │    Codes     │     │
│  └──────────────┘        └──────────────┘     │
│          │                       │             │
│          ▼                       ▼             │
│  ┌──────────────┐        ┌──────────────┐     │
│  │   JSON-RPC   │        │   Utility    │     │
│  │  Responses   │        │  Functions   │     │
│  └──────────────┘        └──────────────┘     │
└────────────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  @neural-trader/mcp   │
        │   (MCP Server with    │
        │    102+ AI Tools)     │
        └───────────────────────┘
```

## API Reference

### JsonRpcRequest

JSON-RPC 2.0 request structure.

```typescript
interface JsonRpcRequest {
  jsonrpc: '2.0';
  method: string;
  params?: any;
  id?: RequestId;  // string | number
}
```

**Example:**

```javascript
const request = {
  jsonrpc: '2.0',
  method: 'run_backtest',
  params: {
    strategy: 'momentum',
    symbol: 'AAPL',
    startDate: '2024-01-01',
    endDate: '2024-12-31'
  },
  id: 'backtest-001'
};
```

### JsonRpcResponse

JSON-RPC 2.0 response structure.

```typescript
interface JsonRpcResponse {
  jsonrpc: '2.0';
  result?: any;
  error?: JsonRpcError;
  id: RequestId;
}
```

**Example:**

```javascript
// Success response
const successResponse = {
  jsonrpc: '2.0',
  result: {
    sharpeRatio: 1.85,
    totalReturn: 0.342,
    maxDrawdown: -0.125
  },
  id: 'backtest-001'
};

// Error response
const errorResponse = {
  jsonrpc: '2.0',
  error: {
    code: -32602,
    message: 'Invalid params: startDate must be before endDate'
  },
  id: 'backtest-001'
};
```

### JsonRpcError

Error object for JSON-RPC 2.0 responses.

```typescript
interface JsonRpcError {
  code: number;
  message: string;
  data?: any;
}
```

**Example:**

```javascript
const error = {
  code: -32603,
  message: 'Internal error: Database connection failed',
  data: {
    timestamp: '2024-01-15T14:30:00Z',
    retryable: true,
    details: 'Connection timeout after 30s'
  }
};
```

### createRequest()

Create a JSON-RPC 2.0 request.

```typescript
function createRequest(
  method: string,
  params?: any,
  id?: RequestId
): JsonRpcRequest
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| method | string | Yes | Method name to invoke |
| params | any | No | Method parameters (object or array) |
| id | string \| number | No | Request ID (omit for notifications) |

**Returns:** `JsonRpcRequest` object

**Example:**

```javascript
const { createRequest } = require('@neural-trader/mcp-protocol');

// Simple request
const simpleRequest = createRequest('list_tools');

// Request with parameters
const paramRequest = createRequest('get_strategy_info', {
  strategyId: 'momentum-v1'
});

// Request with ID for response tracking
const trackedRequest = createRequest(
  'execute_trade',
  {
    symbol: 'AAPL',
    side: 'BUY',
    quantity: 100
  },
  'trade-12345'
);

// Notification (no ID, no response)
const notification = createRequest('price_update', {
  symbol: 'AAPL',
  price: 175.50
});
```

### createSuccessResponse()

Create a successful JSON-RPC 2.0 response.

```typescript
function createSuccessResponse(
  result: any,
  id: RequestId
): JsonRpcResponse
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| result | any | Yes | Result data to return |
| id | string \| number | Yes | Request ID to match |

**Returns:** `JsonRpcResponse` object with result

**Example:**

```javascript
const { createSuccessResponse } = require('@neural-trader/mcp-protocol');

// Simple success response
const response1 = createSuccessResponse({ status: 'ok' }, 'req-001');

// Response with complex data
const response2 = createSuccessResponse(
  {
    portfolio: {
      cash: 50000,
      equity: 125000,
      positions: [
        { symbol: 'AAPL', quantity: 100, value: 17550 },
        { symbol: 'NVDA', quantity: 50, value: 24790 }
      ]
    }
  },
  'portfolio-001'
);

// Response with array data
const response3 = createSuccessResponse(
  ['momentum-v1', 'mean-reversion-v2', 'pairs-trading-v1'],
  'list-strategies-001'
);
```

### createErrorResponse()

Create an error JSON-RPC 2.0 response.

```typescript
function createErrorResponse(
  code: number,
  message: string,
  id: RequestId,
  data?: any
): JsonRpcResponse
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| code | number | Yes | Error code (use ErrorCode constants) |
| message | string | Yes | Human-readable error message |
| id | string \| number | Yes | Request ID to match |
| data | any | No | Additional error details |

**Returns:** `JsonRpcResponse` object with error

**Example:**

```javascript
const { createErrorResponse, ErrorCode } = require('@neural-trader/mcp-protocol');

// Method not found error
const error1 = createErrorResponse(
  ErrorCode.METHOD_NOT_FOUND,
  'Method "invalid_method" does not exist',
  'req-001'
);

// Invalid parameters error with details
const error2 = createErrorResponse(
  ErrorCode.INVALID_PARAMS,
  'Invalid trade parameters',
  'trade-001',
  {
    missing: ['symbol', 'quantity'],
    invalid: { side: 'INVALID_SIDE' }
  }
);

// Internal error with stack trace
const error3 = createErrorResponse(
  ErrorCode.INTERNAL_ERROR,
  'Database query failed',
  'db-query-001',
  {
    query: 'SELECT * FROM trades WHERE ...',
    error: 'Connection timeout',
    timestamp: new Date().toISOString()
  }
);
```

### ErrorCode Constants

Standard JSON-RPC 2.0 error codes.

```typescript
const ErrorCode = {
  PARSE_ERROR: -32700,
  INVALID_REQUEST: -32600,
  METHOD_NOT_FOUND: -32601,
  INVALID_PARAMS: -32602,
  INTERNAL_ERROR: -32603,
  SERVER_ERROR_START: -32099,
  SERVER_ERROR_END: -32000
};
```

**Usage:**

```javascript
const { ErrorCode } = require('@neural-trader/mcp-protocol');

// Use constants instead of magic numbers
if (error.code === ErrorCode.METHOD_NOT_FOUND) {
  console.log('Method does not exist');
}

// Custom server errors (between SERVER_ERROR_END and SERVER_ERROR_START)
const CUSTOM_ERROR_CODE = -32050;
const customError = createErrorResponse(
  CUSTOM_ERROR_CODE,
  'Strategy execution failed',
  'strategy-001'
);
```

## Detailed Tutorials

### Tutorial 1: Building a Request/Response Handler

**Goal:** Create a type-safe request/response handler for trading operations.

**Prerequisites:**
- Node.js 18+
- Basic understanding of JSON-RPC 2.0

**Step 1: Create Request Handler**

```javascript
const {
  createRequest,
  createSuccessResponse,
  createErrorResponse,
  ErrorCode
} = require('@neural-trader/mcp-protocol');

class TradingRequestHandler {
  constructor() {
    this.requestCounter = 0;
  }

  // Generate unique request ID
  generateRequestId() {
    return `req-${Date.now()}-${++this.requestCounter}`;
  }

  // Create a trading request
  createTradeRequest(symbol, side, quantity, orderType) {
    return createRequest(
      'execute_trade',
      {
        symbol,
        side,
        quantity,
        orderType,
        timestamp: new Date().toISOString()
      },
      this.generateRequestId()
    );
  }

  // Create a backtest request
  createBacktestRequest(strategy, config) {
    return createRequest(
      'run_backtest',
      {
        strategy,
        config,
        timestamp: new Date().toISOString()
      },
      this.generateRequestId()
    );
  }

  // Create a query request
  createQueryRequest(method, params) {
    return createRequest(method, params, this.generateRequestId());
  }
}

// Usage
const handler = new TradingRequestHandler();
const tradeRequest = handler.createTradeRequest('AAPL', 'BUY', 100, 'MARKET');
console.log(tradeRequest);
```

**Step 2: Create Response Handler**

```javascript
class TradingResponseHandler {
  // Handle successful trade execution
  handleTradeSuccess(orderId, fillPrice, requestId) {
    return createSuccessResponse(
      {
        orderId,
        status: 'FILLED',
        fillPrice,
        timestamp: new Date().toISOString()
      },
      requestId
    );
  }

  // Handle backtest results
  handleBacktestSuccess(results, requestId) {
    return createSuccessResponse(
      {
        metrics: results.metrics,
        trades: results.trades,
        equityCurve: results.equityCurve
      },
      requestId
    );
  }

  // Handle errors
  handleError(code, message, requestId, details) {
    return createErrorResponse(code, message, requestId, details);
  }

  // Validate and route response
  routeResponse(response) {
    if (response.error) {
      console.error(`Error ${response.error.code}: ${response.error.message}`);
      return { success: false, error: response.error };
    } else {
      console.log('Success:', response.result);
      return { success: true, result: response.result };
    }
  }
}

// Usage
const responseHandler = new TradingResponseHandler();
const successResponse = responseHandler.handleTradeSuccess(
  'order-12345',
  175.50,
  'req-001'
);
console.log(successResponse);
```

**Step 3: Complete Request/Response Cycle**

```javascript
async function completeTradingCycle() {
  const requestHandler = new TradingRequestHandler();
  const responseHandler = new TradingResponseHandler();

  // Create request
  const request = requestHandler.createTradeRequest('AAPL', 'BUY', 100, 'MARKET');
  console.log('Request:', request);

  // Simulate processing (in real app, this would be network call)
  try {
    // Simulate successful trade
    const orderId = 'order-' + Date.now();
    const fillPrice = 175.50;

    const response = responseHandler.handleTradeSuccess(
      orderId,
      fillPrice,
      request.id
    );

    console.log('Response:', response);

    // Route and handle response
    const result = responseHandler.routeResponse(response);
    if (result.success) {
      console.log('Trade executed successfully:', result.result);
    }
  } catch (error) {
    const errorResponse = responseHandler.handleError(
      ErrorCode.INTERNAL_ERROR,
      error.message,
      request.id,
      { error: error.toString() }
    );

    console.log('Error Response:', errorResponse);
  }
}

completeTradingCycle();
```

---

### Tutorial 2: Error Handling Patterns

**Goal:** Implement comprehensive error handling with proper error codes.

**Complete Implementation:**

```javascript
const {
  createErrorResponse,
  ErrorCode
} = require('@neural-trader/mcp-protocol');

class TradingErrorHandler {
  // Parse error - invalid JSON received
  handleParseError(requestId, rawData) {
    return createErrorResponse(
      ErrorCode.PARSE_ERROR,
      'Parse error: Invalid JSON',
      requestId || null,
      {
        receivedData: rawData.substring(0, 100), // First 100 chars
        timestamp: new Date().toISOString()
      }
    );
  }

  // Invalid request - missing required fields
  handleInvalidRequest(requestId, missingFields) {
    return createErrorResponse(
      ErrorCode.INVALID_REQUEST,
      'Invalid Request: Missing required fields',
      requestId || null,
      {
        missing: missingFields,
        requiredFields: ['jsonrpc', 'method'],
        timestamp: new Date().toISOString()
      }
    );
  }

  // Method not found
  handleMethodNotFound(method, requestId) {
    return createErrorResponse(
      ErrorCode.METHOD_NOT_FOUND,
      `Method not found: ${method}`,
      requestId,
      {
        method,
        availableMethods: [
          'execute_trade',
          'run_backtest',
          'get_portfolio',
          'list_strategies'
        ],
        timestamp: new Date().toISOString()
      }
    );
  }

  // Invalid parameters
  handleInvalidParams(method, requestId, validationErrors) {
    return createErrorResponse(
      ErrorCode.INVALID_PARAMS,
      `Invalid params for method: ${method}`,
      requestId,
      {
        method,
        errors: validationErrors,
        timestamp: new Date().toISOString()
      }
    );
  }

  // Internal error
  handleInternalError(requestId, error) {
    return createErrorResponse(
      ErrorCode.INTERNAL_ERROR,
      'Internal error occurred',
      requestId,
      {
        error: error.message,
        stack: process.env.NODE_ENV === 'development' ? error.stack : undefined,
        timestamp: new Date().toISOString()
      }
    );
  }

  // Custom server error
  handleCustomError(code, message, requestId, details) {
    // Ensure code is in server error range
    if (code > ErrorCode.SERVER_ERROR_END || code < ErrorCode.SERVER_ERROR_START) {
      code = ErrorCode.SERVER_ERROR_END - 1; // Default custom error code
    }

    return createErrorResponse(code, message, requestId, details);
  }

  // Validation helper
  validateTradeParams(params) {
    const errors = [];

    if (!params.symbol || typeof params.symbol !== 'string') {
      errors.push({ field: 'symbol', message: 'Required string field' });
    }

    if (!params.side || !['BUY', 'SELL'].includes(params.side)) {
      errors.push({ field: 'side', message: 'Must be BUY or SELL' });
    }

    if (!params.quantity || typeof params.quantity !== 'number' || params.quantity <= 0) {
      errors.push({ field: 'quantity', message: 'Must be positive number' });
    }

    return errors;
  }

  // Process request with error handling
  async processRequest(request) {
    try {
      // Validate request structure
      if (!request.jsonrpc || request.jsonrpc !== '2.0') {
        return this.handleInvalidRequest(request.id, ['jsonrpc']);
      }

      if (!request.method) {
        return this.handleInvalidRequest(request.id, ['method']);
      }

      // Check method exists
      const validMethods = ['execute_trade', 'run_backtest', 'get_portfolio'];
      if (!validMethods.includes(request.method)) {
        return this.handleMethodNotFound(request.method, request.id);
      }

      // Validate parameters
      if (request.method === 'execute_trade') {
        const validationErrors = this.validateTradeParams(request.params || {});
        if (validationErrors.length > 0) {
          return this.handleInvalidParams(request.method, request.id, validationErrors);
        }
      }

      // Process valid request (would call actual implementation)
      // ... processing logic ...

      return null; // No error
    } catch (error) {
      return this.handleInternalError(request.id, error);
    }
  }
}

// Usage example
const errorHandler = new TradingErrorHandler();

// Test various error scenarios
const requests = [
  // Invalid JSON structure
  { invalidData: 'not json' },

  // Missing method
  { jsonrpc: '2.0', id: 'req-001' },

  // Method not found
  { jsonrpc: '2.0', method: 'invalid_method', id: 'req-002' },

  // Invalid parameters
  { jsonrpc: '2.0', method: 'execute_trade', params: { side: 'BUY' }, id: 'req-003' },

  // Valid request
  {
    jsonrpc: '2.0',
    method: 'execute_trade',
    params: { symbol: 'AAPL', side: 'BUY', quantity: 100 },
    id: 'req-004'
  }
];

requests.forEach(async (request) => {
  const error = await errorHandler.processRequest(request);
  if (error) {
    console.log('Error Response:', JSON.stringify(error, null, 2));
  } else {
    console.log('Request valid, processing...');
  }
});
```

---

### Tutorial 3: Client-Server Communication

**Goal:** Implement bidirectional client-server communication with MCP protocol.

**Complete Implementation:**

```javascript
const {
  createRequest,
  createSuccessResponse,
  createErrorResponse,
  ErrorCode
} = require('@neural-trader/mcp-protocol');

class MCPClient {
  constructor() {
    this.pendingRequests = new Map();
    this.requestCounter = 0;
  }

  // Send request and track for response
  async sendRequest(method, params) {
    const id = `client-${Date.now()}-${++this.requestCounter}`;
    const request = createRequest(method, params, id);

    // Create promise that resolves when response received
    return new Promise((resolve, reject) => {
      this.pendingRequests.set(id, { resolve, reject });

      // Send request (in real app, this would be network call)
      this.transport(request);

      // Timeout after 30 seconds
      setTimeout(() => {
        if (this.pendingRequests.has(id)) {
          this.pendingRequests.delete(id);
          reject(new Error('Request timeout'));
        }
      }, 30000);
    });
  }

  // Handle response received
  handleResponse(response) {
    const pending = this.pendingRequests.get(response.id);
    if (!pending) {
      console.warn('Received response for unknown request:', response.id);
      return;
    }

    this.pendingRequests.delete(response.id);

    if (response.error) {
      pending.reject(new Error(response.error.message));
    } else {
      pending.resolve(response.result);
    }
  }

  // Transport layer (placeholder)
  transport(request) {
    // In real implementation, this would send over WebSocket/HTTP
    console.log('Sending request:', request);

    // Simulate response (in real app, response comes from server)
    setTimeout(() => {
      const response = createSuccessResponse(
        { status: 'processed' },
        request.id
      );
      this.handleResponse(response);
    }, 100);
  }
}

class MCPServer {
  constructor() {
    this.methods = new Map();
    this.registerDefaultMethods();
  }

  // Register method handler
  registerMethod(name, handler) {
    this.methods.set(name, handler);
  }

  // Register default trading methods
  registerDefaultMethods() {
    this.registerMethod('execute_trade', async (params) => {
      return {
        orderId: 'order-' + Date.now(),
        status: 'FILLED',
        fillPrice: 175.50
      };
    });

    this.registerMethod('get_portfolio', async (params) => {
      return {
        cash: 50000,
        equity: 125000,
        positions: [
          { symbol: 'AAPL', quantity: 100 }
        ]
      };
    });

    this.registerMethod('list_strategies', async (params) => {
      return {
        strategies: ['momentum-v1', 'mean-reversion-v2']
      };
    });
  }

  // Handle incoming request
  async handleRequest(request) {
    try {
      // Validate request
      if (!request.jsonrpc || request.jsonrpc !== '2.0') {
        return createErrorResponse(
          ErrorCode.INVALID_REQUEST,
          'Invalid JSON-RPC version',
          request.id || null
        );
      }

      // Check if method exists
      if (!this.methods.has(request.method)) {
        return createErrorResponse(
          ErrorCode.METHOD_NOT_FOUND,
          `Method not found: ${request.method}`,
          request.id,
          { availableMethods: Array.from(this.methods.keys()) }
        );
      }

      // Execute method
      const handler = this.methods.get(request.method);
      const result = await handler(request.params || {});

      // Return success response (or null for notifications)
      return request.id ? createSuccessResponse(result, request.id) : null;
    } catch (error) {
      return createErrorResponse(
        ErrorCode.INTERNAL_ERROR,
        error.message,
        request.id || null,
        { error: error.toString() }
      );
    }
  }
}

// Usage example: Complete client-server cycle
async function demonstrateClientServer() {
  const client = new MCPClient();
  const server = new MCPServer();

  // Override client transport to actually call server
  client.transport = async (request) => {
    console.log('Client → Server:', request);

    const response = await server.handleRequest(request);
    if (response) {
      console.log('Server → Client:', response);
      client.handleResponse(response);
    }
  };

  // Execute requests
  try {
    // Request 1: Execute trade
    const tradeResult = await client.sendRequest('execute_trade', {
      symbol: 'AAPL',
      side: 'BUY',
      quantity: 100
    });
    console.log('Trade Result:', tradeResult);

    // Request 2: Get portfolio
    const portfolio = await client.sendRequest('get_portfolio', {});
    console.log('Portfolio:', portfolio);

    // Request 3: List strategies
    const strategies = await client.sendRequest('list_strategies', {});
    console.log('Strategies:', strategies);

    // Request 4: Invalid method (error)
    try {
      await client.sendRequest('invalid_method', {});
    } catch (error) {
      console.log('Expected error:', error.message);
    }
  } catch (error) {
    console.error('Request failed:', error);
  }
}

demonstrateClientServer();
```

---

### Tutorial 4: Batch Request Processing

**Goal:** Process multiple requests efficiently in a single batch.

**Implementation:**

```javascript
const {
  createRequest,
  createSuccessResponse,
  createErrorResponse,
  ErrorCode
} = require('@neural-trader/mcp-protocol');

class BatchRequestProcessor {
  // Create batch of requests
  createBatch(requests) {
    return requests.map((req, index) =>
      createRequest(req.method, req.params, `batch-${index}`)
    );
  }

  // Process batch of requests
  async processBatch(requests) {
    const responses = [];

    for (const request of requests) {
      try {
        // Process each request
        const result = await this.processRequest(request);
        responses.push(createSuccessResponse(result, request.id));
      } catch (error) {
        responses.push(
          createErrorResponse(
            ErrorCode.INTERNAL_ERROR,
            error.message,
            request.id
          )
        );
      }
    }

    return responses;
  }

  // Single request processor
  async processRequest(request) {
    // Simulate processing
    switch (request.method) {
      case 'get_price':
        return { symbol: request.params.symbol, price: 175.50 };
      case 'get_portfolio':
        return { cash: 50000, equity: 125000 };
      case 'list_strategies':
        return { strategies: ['momentum-v1', 'mean-reversion-v2'] };
      default:
        throw new Error('Method not found');
    }
  }
}

// Usage
async function demonstrateBatchProcessing() {
  const processor = new BatchRequestProcessor();

  // Create batch of requests
  const batch = processor.createBatch([
    { method: 'get_price', params: { symbol: 'AAPL' } },
    { method: 'get_price', params: { symbol: 'NVDA' } },
    { method: 'get_portfolio', params: {} },
    { method: 'list_strategies', params: {} }
  ]);

  console.log('Batch requests:', batch);

  // Process batch
  const responses = await processor.processBatch(batch);
  console.log('Batch responses:', responses);
}

demonstrateBatchProcessing();
```

---

### Tutorial 5: Type-Safe Protocol Validation

**Goal:** Validate JSON-RPC 2.0 messages with TypeScript.

**Advanced Example:**

```typescript
import {
  JsonRpcRequest,
  JsonRpcResponse,
  JsonRpcError,
  RequestId
} from '@neural-trader/mcp-protocol';

class ProtocolValidator {
  // Validate request structure
  validateRequest(data: unknown): JsonRpcRequest | null {
    if (!this.isObject(data)) return null;

    const request = data as Partial<JsonRpcRequest>;

    // Required fields
    if (request.jsonrpc !== '2.0') return null;
    if (typeof request.method !== 'string') return null;

    // Optional fields
    if (request.params !== undefined && !this.isValidParams(request.params)) {
      return null;
    }

    if (request.id !== undefined && !this.isValidId(request.id)) {
      return null;
    }

    return request as JsonRpcRequest;
  }

  // Validate response structure
  validateResponse(data: unknown): JsonRpcResponse | null {
    if (!this.isObject(data)) return null;

    const response = data as Partial<JsonRpcResponse>;

    // Required fields
    if (response.jsonrpc !== '2.0') return null;
    if (!this.isValidId(response.id)) return null;

    // Must have either result or error, not both
    const hasResult = response.result !== undefined;
    const hasError = response.error !== undefined;

    if ((hasResult && hasError) || (!hasResult && !hasError)) {
      return null;
    }

    if (hasError && !this.isValidError(response.error)) {
      return null;
    }

    return response as JsonRpcResponse;
  }

  // Validate error object
  validateError(data: unknown): JsonRpcError | null {
    if (!this.isObject(data)) return null;

    const error = data as Partial<JsonRpcError>;

    if (typeof error.code !== 'number') return null;
    if (typeof error.message !== 'string') return null;

    return error as JsonRpcError;
  }

  // Helper methods
  private isObject(value: unknown): boolean {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
  }

  private isValidParams(params: unknown): boolean {
    return typeof params === 'object' && params !== null;
  }

  private isValidId(id: unknown): id is RequestId {
    return typeof id === 'string' || typeof id === 'number';
  }

  private isValidError(error: unknown): boolean {
    if (!this.isObject(error)) return false;

    const err = error as Partial<JsonRpcError>;
    return typeof err.code === 'number' && typeof err.message === 'string';
  }
}

// Usage
const validator = new ProtocolValidator();

// Valid request
const validRequest = {
  jsonrpc: '2.0',
  method: 'execute_trade',
  params: { symbol: 'AAPL' },
  id: 'req-001'
};

const request = validator.validateRequest(validRequest);
if (request) {
  console.log('Valid request:', request);
} else {
  console.log('Invalid request');
}

// Valid response
const validResponse = {
  jsonrpc: '2.0',
  result: { orderId: 'order-001' },
  id: 'req-001'
};

const response = validator.validateResponse(validResponse);
if (response) {
  console.log('Valid response:', response);
} else {
  console.log('Invalid response');
}
```

## Integration Examples

### Integration with @neural-trader/mcp

The MCP server uses mcp-protocol for all communication.

```javascript
const { McpServer } = require('@neural-trader/mcp');
const {
  createRequest,
  createSuccessResponse,
  ErrorCode
} = require('@neural-trader/mcp-protocol');

// MCP server automatically handles protocol
const server = new McpServer({ transport: 'stdio' });

// All requests/responses use mcp-protocol types
await server.start();
```

### Integration with @neural-trader/core

Protocol works seamlessly with core trading types.

```javascript
const {
  createRequest,
  createSuccessResponse
} = require('@neural-trader/mcp-protocol');
const { BacktestConfig } = require('@neural-trader/core');

// Create backtest request with core types
const backtestConfig = {
  initialCapital: 100000,
  startDate: '2024-01-01',
  endDate: '2024-12-31',
  commission: 0.001,
  slippage: 0.0005,
  useMarkToMarket: true
};

const request = createRequest('run_backtest', backtestConfig, 'backtest-001');

// Response also uses core types
const response = createSuccessResponse(
  {
    metrics: { /* BacktestMetrics */ },
    trades: [ /* Trade[] */ ]
  },
  'backtest-001'
);
```

### Custom Transport Layer

Use protocol with any transport (WebSocket, HTTP, stdio).

```javascript
const {
  createRequest,
  createSuccessResponse
} = require('@neural-trader/mcp-protocol');

class WebSocketTransport {
  constructor(url) {
    this.ws = new WebSocket(url);
    this.ws.onmessage = (event) => this.handleMessage(event.data);
  }

  send(method, params, id) {
    const request = createRequest(method, params, id);
    this.ws.send(JSON.stringify(request));
  }

  handleMessage(data) {
    const response = JSON.parse(data);
    // Process response...
  }
}
```

## Related Packages

### Core Packages

- **[@neural-trader/core](https://www.npmjs.com/package/@neural-trader/core)** - Core types and interfaces
- **[@neural-trader/mcp-protocol](https://www.npmjs.com/package/@neural-trader/mcp-protocol)** - JSON-RPC 2.0 protocol (this package)
- **[@neural-trader/mcp](https://www.npmjs.com/package/@neural-trader/mcp)** - MCP server with 102+ AI tools

### Recommended Combinations

**For AI Assistant Integration:**
```bash
npm install @neural-trader/mcp-protocol @neural-trader/mcp
```

**For Custom MCP Server:**
```bash
npm install @neural-trader/mcp-protocol @neural-trader/core
```

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

This project is dual-licensed under **MIT OR Apache-2.0**.

- **MIT License**: See [LICENSE-MIT](../../LICENSE-MIT)
- **Apache License 2.0**: See [LICENSE-APACHE](../../LICENSE-APACHE)

## Support

- **GitHub Issues**: https://github.com/ruvnet/neural-trader/issues
- **Discord**: https://discord.gg/neural-trader
- **Email**: support@neural-trader.io

## Specifications

- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)
- [Model Context Protocol](https://anthropic.com/docs/mcp)

---

**Disclaimer**: This software is for educational and research purposes only. Trading financial instruments carries risk. Use at your own risk.
