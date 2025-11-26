/**
 * Rate Limiting Middleware for MCP Tools
 *
 * Implements token bucket algorithm with Redis backend for distributed rate limiting.
 * Prevents API abuse and ensures fair resource allocation across clients.
 *
 * @module middleware/rate-limiter
 */

const Redis = require('ioredis');

/**
 * Rate limiter configuration
 * @typedef {Object} RateLimiterConfig
 * @property {number} windowMs - Time window in milliseconds
 * @property {number} maxRequests - Maximum requests per window
 * @property {string} keyPrefix - Redis key prefix
 * @property {boolean} skipFailedRequests - Don't count failed requests
 */

class RateLimiter {
  /**
   * @param {RateLimiterConfig} config
   */
  constructor(config = {}) {
    this.windowMs = config.windowMs || 60000; // 1 minute default
    this.maxRequests = config.maxRequests || 100;
    this.keyPrefix = config.keyPrefix || 'ratelimit:';
    this.skipFailedRequests = config.skipFailedRequests || false;

    // Connect to Redis (or use in-memory fallback)
    this.redis = this._connectRedis(config.redisUrl);
    this.useRedis = !!this.redis;

    // In-memory fallback for development
    this.memoryStore = new Map();
  }

  /**
   * Connect to Redis server
   * @private
   */
  _connectRedis(redisUrl) {
    try {
      const url = redisUrl || process.env.REDIS_URL || 'redis://localhost:6379';
      const redis = new Redis(url, {
        retryStrategy: (times) => {
          if (times > 3) {
            console.warn('⚠️  Redis connection failed, using in-memory rate limiting');
            return null;
          }
          return Math.min(times * 50, 2000);
        },
        maxRetriesPerRequest: 3,
      });

      redis.on('error', (err) => {
        console.warn('⚠️  Redis error:', err.message);
      });

      return redis;
    } catch (error) {
      console.warn('⚠️  Failed to connect to Redis, using in-memory fallback');
      return null;
    }
  }

  /**
   * Check rate limit for a client
   * @param {string} clientId - Unique client identifier
   * @returns {Promise<{allowed: boolean, remaining: number, resetTime: number}>}
   */
  async checkLimit(clientId) {
    if (this.useRedis && this.redis) {
      return this._checkLimitRedis(clientId);
    }
    return this._checkLimitMemory(clientId);
  }

  /**
   * Redis-based rate limiting (distributed)
   * @private
   */
  async _checkLimitRedis(clientId) {
    const key = `${this.keyPrefix}${clientId}`;
    const now = Date.now();
    const windowStart = now - this.windowMs;

    try {
      // Use Redis sorted set for sliding window
      const multi = this.redis.multi();

      // Remove old entries outside the window
      multi.zremrangebyscore(key, 0, windowStart);

      // Count requests in current window
      multi.zcard(key);

      // Add current request
      multi.zadd(key, now, `${now}-${Math.random()}`);

      // Set expiration
      multi.expire(key, Math.ceil(this.windowMs / 1000));

      const results = await multi.exec();
      const count = results[1][1]; // zcard result

      const allowed = count < this.maxRequests;
      const remaining = Math.max(0, this.maxRequests - count - 1);
      const resetTime = now + this.windowMs;

      return { allowed, remaining, resetTime };
    } catch (error) {
      console.error('Rate limit check failed:', error);
      // Fail open - allow request on error
      return { allowed: true, remaining: this.maxRequests, resetTime: now + this.windowMs };
    }
  }

  /**
   * In-memory rate limiting (single instance)
   * @private
   */
  async _checkLimitMemory(clientId) {
    const now = Date.now();
    const windowStart = now - this.windowMs;

    // Get or create client entry
    let clientData = this.memoryStore.get(clientId);
    if (!clientData) {
      clientData = { requests: [], resetTime: now + this.windowMs };
      this.memoryStore.set(clientId, clientData);
    }

    // Remove old requests outside window
    clientData.requests = clientData.requests.filter((timestamp) => timestamp > windowStart);

    // Check limit
    const allowed = clientData.requests.length < this.maxRequests;
    if (allowed) {
      clientData.requests.push(now);
    }

    const remaining = Math.max(0, this.maxRequests - clientData.requests.length - (allowed ? 1 : 0));
    const resetTime = now + this.windowMs;
    clientData.resetTime = resetTime;

    // Cleanup old entries
    this._cleanupMemoryStore(windowStart);

    return { allowed, remaining, resetTime };
  }

  /**
   * Cleanup old memory store entries
   * @private
   */
  _cleanupMemoryStore(windowStart) {
    // Run cleanup every 100 requests
    if (Math.random() < 0.01) {
      for (const [clientId, data] of this.memoryStore.entries()) {
        if (data.requests.length === 0 || Math.max(...data.requests) < windowStart) {
          this.memoryStore.delete(clientId);
        }
      }
    }
  }

  /**
   * Express/Connect middleware
   */
  middleware() {
    return async (req, res, next) => {
      // Extract client ID (IP address or API key)
      const clientId = req.headers['x-api-key'] || req.ip || req.connection.remoteAddress || 'unknown';

      const result = await this.checkLimit(clientId);

      // Set rate limit headers
      res.setHeader('X-RateLimit-Limit', this.maxRequests);
      res.setHeader('X-RateLimit-Remaining', result.remaining);
      res.setHeader('X-RateLimit-Reset', new Date(result.resetTime).toISOString());

      if (!result.allowed) {
        res.status(429).json({
          error: 'Too Many Requests',
          message: `Rate limit exceeded. Maximum ${this.maxRequests} requests per ${this.windowMs / 1000} seconds.`,
          retryAfter: Math.ceil((result.resetTime - Date.now()) / 1000),
        });
        return;
      }

      next();
    };
  }

  /**
   * Close Redis connection
   */
  async close() {
    if (this.redis) {
      await this.redis.quit();
    }
  }
}

/**
 * Create rate limiter instances for different tool categories
 */
const rateLimiters = {
  // Default rate limit: 100 requests/minute
  default: new RateLimiter({ maxRequests: 100, windowMs: 60000 }),

  // Odds API: 50 requests/minute (external API has limits)
  odds: new RateLimiter({ maxRequests: 50, windowMs: 60000, keyPrefix: 'ratelimit:odds:' }),

  // Neural networks: 20 requests/minute (expensive operations)
  neural: new RateLimiter({ maxRequests: 20, windowMs: 60000, keyPrefix: 'ratelimit:neural:' }),

  // E2B sandboxes: 30 requests/minute (resource intensive)
  e2b: new RateLimiter({ maxRequests: 30, windowMs: 60000, keyPrefix: 'ratelimit:e2b:' }),

  // Sports betting: 60 requests/minute
  sports: new RateLimiter({ maxRequests: 60, windowMs: 60000, keyPrefix: 'ratelimit:sports:' }),

  // Authentication: 5 requests/minute (prevent brute force)
  auth: new RateLimiter({ maxRequests: 5, windowMs: 60000, keyPrefix: 'ratelimit:auth:' }),
};

/**
 * Get appropriate rate limiter for a tool
 * @param {string} toolName - MCP tool name
 */
function getRateLimiterForTool(toolName) {
  if (toolName.includes('odds') || toolName.includes('arbitrage')) {
    return rateLimiters.odds;
  }
  if (toolName.includes('neural') || toolName.includes('forecast') || toolName.includes('train')) {
    return rateLimiters.neural;
  }
  if (toolName.includes('e2b') || toolName.includes('sandbox')) {
    return rateLimiters.e2b;
  }
  if (toolName.includes('sports') || toolName.includes('betting')) {
    return rateLimiters.sports;
  }
  if (toolName.includes('auth') || toolName.includes('login') || toolName.includes('register')) {
    return rateLimiters.auth;
  }
  return rateLimiters.default;
}

/**
 * Rate limit wrapper for MCP tool handlers
 * @param {string} toolName - Tool name
 * @param {Function} handler - Tool handler function
 */
function withRateLimit(toolName, handler) {
  const limiter = getRateLimiterForTool(toolName);

  return async (params, context) => {
    // Extract client ID from context
    const clientId = context?.apiKey || context?.sessionId || 'anonymous';

    // Check rate limit
    const result = await limiter.checkLimit(clientId);

    if (!result.allowed) {
      throw new Error(
        `Rate limit exceeded for ${toolName}. ` +
          `Maximum ${limiter.maxRequests} requests per ${limiter.windowMs / 1000} seconds. ` +
          `Retry after ${Math.ceil((result.resetTime - Date.now()) / 1000)} seconds.`
      );
    }

    // Execute handler
    try {
      const response = await handler(params, context);
      return response;
    } catch (error) {
      // Don't count failed requests if configured
      if (limiter.skipFailedRequests) {
        // TODO: Implement request rollback in Redis
      }
      throw error;
    }
  };
}

module.exports = {
  RateLimiter,
  rateLimiters,
  getRateLimiterForTool,
  withRateLimit,
};
