/**
 * Error Handling Middleware Integration
 * Week 2 Optimization: Integration with MCP server and existing middleware
 *
 * This module integrates the error handling framework with:
 * - MCP server tool execution
 * - External API calls (Alpaca, The Odds API, etc.)
 * - Database operations
 * - Redis operations
 */

const {
  errorHandler,
  ExternalAPIError,
  DatabaseError,
  NetworkError,
  RetryStrategy,
} = require('./error-handling');

/**
 * Wrap external API calls with error handling
 */
function withExternalAPIProtection(apiName, fn, options = {}) {
  return async (...args) => {
    const serviceName = `external-api:${apiName}`;

    return errorHandler.executeWithProtection(fn.bind(null, ...args), {
      serviceName,
      retryStrategy: new RetryStrategy({
        maxRetries: options.maxRetries || 3,
        initialDelay: options.initialDelay || 1000,
        maxDelay: options.maxDelay || 10000,
      }),
      circuitBreakerOptions: {
        failureThreshold: options.failureThreshold || 5,
        timeout: options.timeout || 60000,
      },
      fallback: options.fallback,
      context: { apiName, args },
    });
  };
}

/**
 * Wrap database operations with error handling
 */
function withDatabaseProtection(operation, fn, options = {}) {
  return async (...args) => {
    const serviceName = `database:${operation}`;

    return errorHandler.executeWithProtection(fn.bind(null, ...args), {
      serviceName,
      retryStrategy: new RetryStrategy({
        maxRetries: options.maxRetries || 3,
        initialDelay: options.initialDelay || 500,
        maxDelay: options.maxDelay || 5000,
      }),
      circuitBreakerOptions: {
        failureThreshold: options.failureThreshold || 10,
        timeout: options.timeout || 30000,
      },
      context: { operation, args },
    });
  };
}

/**
 * Wrap MCP tool execution with error handling
 */
function withToolErrorHandling(toolName, handler) {
  return async (params) => {
    const serviceName = `mcp-tool:${toolName}`;

    try {
      return await errorHandler.executeWithProtection(handler.bind(null, params), {
        serviceName,
        retryStrategy: new RetryStrategy({
          maxRetries: 2,
          initialDelay: 1000,
          maxDelay: 5000,
        }),
        circuitBreakerOptions: {
          failureThreshold: 5,
          timeout: 60000,
        },
        context: { toolName, params },
      });
    } catch (error) {
      // Transform error to MCP-compatible format
      return {
        error: {
          code: error.code || 'TOOL_EXECUTION_FAILED',
          message: error.message,
          severity: error.severity,
          category: error.category,
          retryable: error.retryable,
        },
      };
    }
  };
}

/**
 * Express/Fastify error middleware
 */
function errorMiddleware(err, req, res, next) {
  // Log error with context
  console.error('❌ Request error:', {
    error: err.message,
    code: err.code,
    severity: err.severity,
    category: err.category,
    path: req.path,
    method: req.method,
    timestamp: new Date().toISOString(),
  });

  // Determine status code
  let statusCode = 500;

  if (err.name === 'ValidationError') {
    statusCode = 400;
  } else if (err.name === 'AuthenticationError') {
    statusCode = 401;
  } else if (err.name === 'AuthorizationError') {
    statusCode = 403;
  } else if (err.name === 'RateLimitError') {
    statusCode = 429;
  } else if (err.name === 'ExternalAPIError') {
    statusCode = 502;
  }

  // Send error response
  res.status(statusCode).json({
    error: {
      message: err.message,
      code: err.code,
      category: err.category,
      severity: err.severity,
    },
  });
}

/**
 * Create fallback for external APIs
 */
function createApiFallback(apiName, defaultValue) {
  return () => {
    console.warn(`⚠️  Using fallback for ${apiName}`);
    return defaultValue;
  };
}

/**
 * Alpaca API with error handling
 */
function createProtectedAlpacaClient(alpaca) {
  return {
    getAccount: withExternalAPIProtection(
      'alpaca:getAccount',
      () => alpaca.getAccount(),
      {
        maxRetries: 3,
        fallback: createApiFallback('alpaca:getAccount', {
          equity: 0,
          buying_power: 0,
          cash: 0,
        }),
      }
    ),

    getPositions: withExternalAPIProtection(
      'alpaca:getPositions',
      () => alpaca.getPositions(),
      {
        maxRetries: 2,
        fallback: createApiFallback('alpaca:getPositions', []),
      }
    ),

    placeOrder: withExternalAPIProtection(
      'alpaca:placeOrder',
      (orderParams) => alpaca.createOrder(orderParams),
      {
        maxRetries: 1, // Only retry once for order placement
        failureThreshold: 3,
      }
    ),

    getBars: withExternalAPIProtection(
      'alpaca:getBars',
      (symbol, timeframe, options) => alpaca.getBars(symbol, timeframe, options),
      {
        maxRetries: 3,
        fallback: (symbol) => createApiFallback(`alpaca:getBars:${symbol}`, []),
      }
    ),
  };
}

/**
 * The Odds API with error handling
 */
function createProtectedOddsClient(oddsClient) {
  return {
    getSports: withExternalAPIProtection(
      'odds-api:getSports',
      () => oddsClient.getSports(),
      {
        maxRetries: 2,
        fallback: createApiFallback('odds-api:getSports', []),
      }
    ),

    getOdds: withExternalAPIProtection(
      'odds-api:getOdds',
      (sport, options) => oddsClient.getOdds(sport, options),
      {
        maxRetries: 2,
        timeout: 30000,
        fallback: createApiFallback('odds-api:getOdds', []),
      }
    ),

    getEventOdds: withExternalAPIProtection(
      'odds-api:getEventOdds',
      (sport, eventId) => oddsClient.getEventOdds(sport, eventId),
      {
        maxRetries: 2,
        fallback: createApiFallback('odds-api:getEventOdds', null),
      }
    ),
  };
}

/**
 * Database pool with error handling
 */
function createProtectedDatabasePool(pool) {
  return {
    query: withDatabaseProtection(
      'query',
      (text, params) => pool.query(text, params),
      { maxRetries: 3 }
    ),

    getClient: withDatabaseProtection(
      'getClient',
      () => pool.getClient(),
      { maxRetries: 2 }
    ),
  };
}

/**
 * Get error handler statistics endpoint
 */
function getErrorStats() {
  return errorHandler.getAllStats();
}

/**
 * Reset all circuit breakers (admin endpoint)
 */
function resetAllCircuitBreakers() {
  const stats = errorHandler.getAllStats();

  for (const [serviceName] of Object.entries(stats.circuitBreakers)) {
    const breaker = errorHandler.getCircuitBreaker(serviceName);
    breaker.reset();
  }

  return { message: 'All circuit breakers reset', count: Object.keys(stats.circuitBreakers).length };
}

module.exports = {
  withExternalAPIProtection,
  withDatabaseProtection,
  withToolErrorHandling,
  errorMiddleware,
  createProtectedAlpacaClient,
  createProtectedOddsClient,
  createProtectedDatabasePool,
  getErrorStats,
  resetAllCircuitBreakers,
};
