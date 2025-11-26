# Express.js Integration Guide

## Overview

This guide demonstrates how to integrate `@neural-trader/backend` with Express.js applications for building high-performance trading APIs.

## Platform Requirements

- **Node.js**: >= 16.0.0 (Recommended: v18 LTS or v20 LTS)
- **Operating Systems**: Linux (x64, ARM64), macOS (Intel, Apple Silicon), Windows (x64)
- **Express.js**: >= 4.18.0
- **Architecture**: Automatic platform detection via NAPI-RS

## Installation

```bash
npm install @neural-trader/backend express
npm install --save-dev @types/express  # For TypeScript projects
```

## Basic Express Server

```javascript
const express = require('express');
const backend = require('@neural-trader/backend');

const app = express();
app.use(express.json());

// Initialize neural-trader backend
let initialized = false;

async function initializeBackend() {
  if (!initialized) {
    // Initialize authentication
    const jwtSecret = process.env.JWT_SECRET;
    if (!jwtSecret) {
      throw new Error('JWT_SECRET environment variable is required');
    }
    backend.initAuth(jwtSecret);

    // Initialize rate limiter
    backend.initRateLimiter({
      maxRequestsPerMinute: 100,
      burstSize: 20,
      windowDurationSecs: 60
    });

    // Initialize audit logger
    backend.initAuditLogger(10000, true, true);

    // Initialize neural trader
    await backend.initNeuralTrader(JSON.stringify({
      logLevel: 'info',
      enableGpu: false
    }));

    initialized = true;
    console.log('Neural Trader backend initialized');
  }
}

// Health check endpoint
app.get('/health', async (req, res) => {
  try {
    const health = await backend.healthCheck();
    res.json(health);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// System info endpoint
app.get('/api/system/info', (req, res) => {
  try {
    const info = backend.getSystemInfo();
    res.json(info);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Start server
const PORT = process.env.PORT || 3000;

initializeBackend()
  .then(() => {
    app.listen(PORT, () => {
      console.log(`Server running on port ${PORT}`);
    });
  })
  .catch(error => {
    console.error('Failed to initialize backend:', error);
    process.exit(1);
  });

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('SIGTERM received, shutting down gracefully');
  await backend.shutdown();
  process.exit(0);
});
```

## Middleware Integration

### Authentication Middleware

```javascript
const authMiddleware = (req, res, next) => {
  const apiKey = req.headers['x-api-key'];

  if (!apiKey) {
    backend.logAuditEvent(
      'Warning',
      'Authentication',
      'missing_api_key',
      'failure',
      null,
      null,
      req.ip
    );
    return res.status(401).json({ error: 'API key required' });
  }

  try {
    const user = backend.validateApiKey(apiKey);
    req.user = user;

    backend.logAuditEvent(
      'Info',
      'Authentication',
      'api_key_validated',
      'success',
      user.userId,
      user.username,
      req.ip
    );

    next();
  } catch (error) {
    backend.logAuditEvent(
      'Security',
      'Authentication',
      'invalid_api_key',
      'failure',
      null,
      null,
      req.ip
    );
    res.status(401).json({ error: 'Invalid API key' });
  }
};
```

### Rate Limiting Middleware

```javascript
const rateLimitMiddleware = (req, res, next) => {
  const identifier = req.user?.userId || req.ip;

  if (!backend.checkRateLimit(identifier, 1)) {
    backend.logAuditEvent(
      'Warning',
      'Security',
      'rate_limit_exceeded',
      'failure',
      req.user?.userId,
      req.user?.username,
      req.ip
    );

    const stats = backend.getRateLimitStats(identifier);
    return res.status(429).json({
      error: 'Rate limit exceeded',
      retryAfter: Math.ceil((60 - stats.totalRequests % 60) / stats.refillRate),
      stats
    });
  }

  next();
};
```

### Security Headers Middleware

```javascript
const securityHeadersMiddleware = (req, res, next) => {
  const headers = JSON.parse(backend.getSecurityHeaders());

  for (const [key, value] of Object.entries(headers)) {
    res.setHeader(key, value);
  }

  next();
};
```

### CORS Middleware

```javascript
const corsMiddleware = (req, res, next) => {
  const origin = req.headers.origin;

  if (backend.checkCorsOrigin(origin || '*')) {
    const corsHeaders = JSON.parse(backend.getCorsHeaders(origin));

    for (const [key, value] of Object.entries(corsHeaders)) {
      res.setHeader(key, value);
    }
  }

  next();
};
```

### Input Sanitization Middleware

```javascript
const sanitizeMiddleware = (req, res, next) => {
  if (req.body) {
    for (const [key, value] of Object.entries(req.body)) {
      if (typeof value === 'string') {
        const sanitized = backend.sanitizeInput(value);
        const threats = backend.checkSecurityThreats(sanitized);

        if (threats.length > 0) {
          backend.logAuditEvent(
            'Critical',
            'Security',
            'security_threat_detected',
            'failure',
            req.user?.userId,
            req.user?.username,
            req.ip,
            key,
            JSON.stringify({ threats, value: sanitized.substring(0, 100) })
          );

          return res.status(400).json({
            error: 'Security threat detected in input',
            threats
          });
        }

        req.body[key] = sanitized;
      }
    }
  }

  next();
};
```

## Trading API Endpoints

### Market Analysis

```javascript
app.get('/api/analysis/:symbol', authMiddleware, rateLimitMiddleware, async (req, res) => {
  try {
    const { symbol } = req.params;
    const { useGpu = false } = req.query;

    const analysis = await backend.quickAnalysis(symbol, useGpu);

    backend.logAuditEvent(
      'Info',
      'DataAccess',
      'market_analysis',
      'success',
      req.user.userId,
      req.user.username,
      req.ip,
      symbol
    );

    res.json(analysis);
  } catch (error) {
    backend.logAuditEvent(
      'Error',
      'DataAccess',
      'market_analysis',
      'failure',
      req.user.userId,
      req.user.username,
      req.ip,
      req.params.symbol,
      JSON.stringify({ error: error.message })
    );

    res.status(500).json({ error: error.message });
  }
});
```

### Trading Execution

```javascript
app.post('/api/trade/execute',
  authMiddleware,
  rateLimitMiddleware,
  sanitizeMiddleware,
  async (req, res) => {
    try {
      const { strategy, symbol, action, quantity, orderType, limitPrice } = req.body;

      // Validate trading parameters
      if (!backend.validateTradingParams(symbol, quantity, limitPrice)) {
        return res.status(400).json({ error: 'Invalid trading parameters' });
      }

      // Check authorization
      const authorized = backend.checkAuthorization(
        req.headers['x-api-key'],
        'execute_trade',
        'User'
      );

      if (!authorized) {
        backend.logAuditEvent(
          'Security',
          'Authorization',
          'unauthorized_trade',
          'failure',
          req.user.userId,
          req.user.username,
          req.ip,
          symbol
        );

        return res.status(403).json({ error: 'Insufficient permissions' });
      }

      const result = await backend.executeTrade(
        strategy,
        symbol,
        action,
        quantity,
        orderType,
        limitPrice
      );

      backend.logAuditEvent(
        'Info',
        'Trading',
        'trade_executed',
        'success',
        req.user.userId,
        req.user.username,
        req.ip,
        symbol,
        JSON.stringify({ orderId: result.orderId, action, quantity })
      );

      res.json(result);
    } catch (error) {
      backend.logAuditEvent(
        'Error',
        'Trading',
        'trade_execution_failed',
        'failure',
        req.user.userId,
        req.user.username,
        req.ip,
        req.body.symbol,
        JSON.stringify({ error: error.message })
      );

      res.status(500).json({ error: error.message });
    }
  }
);
```

### Backtesting

```javascript
app.post('/api/backtest', authMiddleware, rateLimitMiddleware, async (req, res) => {
  try {
    const { strategy, symbol, startDate, endDate, useGpu = false } = req.body;

    const result = await backend.runBacktest(
      strategy,
      symbol,
      startDate,
      endDate,
      useGpu
    );

    backend.logAuditEvent(
      'Info',
      'DataAccess',
      'backtest_executed',
      'success',
      req.user.userId,
      req.user.username,
      req.ip,
      symbol,
      JSON.stringify({ strategy, startDate, endDate })
    );

    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
```

### Portfolio Status

```javascript
app.get('/api/portfolio', authMiddleware, rateLimitMiddleware, async (req, res) => {
  try {
    const { includeAnalytics = true } = req.query;

    const status = await backend.getPortfolioStatus(includeAnalytics);

    res.json(status);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
```

## Advanced Features

### E2B Swarm Integration

```javascript
app.post('/api/swarm/init', authMiddleware, async (req, res) => {
  try {
    const { topology, config } = req.body;

    const result = await backend.initE2bSwarm(topology, JSON.stringify(config));

    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/swarm/:swarmId/status', authMiddleware, async (req, res) => {
  try {
    const { swarmId } = req.params;

    const status = await backend.getSwarmStatus(swarmId);

    res.json(status);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
```

### Sports Betting Syndicate

```javascript
app.post('/api/syndicate/create', authMiddleware, async (req, res) => {
  try {
    const { syndicateId, name, description } = req.body;

    const syndicate = await backend.createSyndicate(syndicateId, name, description);

    backend.logAuditEvent(
      'Info',
      'Configuration',
      'syndicate_created',
      'success',
      req.user.userId,
      req.user.username,
      req.ip,
      syndicateId
    );

    res.json(syndicate);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
```

## Error Handling

```javascript
// Global error handler
app.use((err, req, res, next) => {
  backend.logAuditEvent(
    'Error',
    'System',
    'unhandled_error',
    'failure',
    req.user?.userId,
    req.user?.username,
    req.ip,
    req.path,
    JSON.stringify({
      error: err.message,
      stack: err.stack?.substring(0, 500)
    })
  );

  res.status(500).json({
    error: 'Internal server error',
    requestId: req.id || Date.now().toString()
  });
});
```

## Environment Configuration

Create a `.env` file:

```env
# Required
JWT_SECRET=your-64-byte-secret-here
NODE_ENV=production

# Server
PORT=3000
HOST=0.0.0.0

# Rate Limiting
RATE_LIMIT_MAX_REQUESTS=100
RATE_LIMIT_WINDOW_SECS=60

# CORS
CORS_ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# Security
REQUIRE_HTTPS=true
ENABLE_AUDIT_LOGGING=true
```

## Testing

```javascript
// test/integration.test.js
const request = require('supertest');
const app = require('../app');

describe('Trading API', () => {
  let apiKey;

  beforeAll(() => {
    // Create test API key
    apiKey = backend.createApiKey('test-user', 'User', 1000, 365);
  });

  test('GET /api/analysis/:symbol returns market analysis', async () => {
    const response = await request(app)
      .get('/api/analysis/AAPL')
      .set('x-api-key', apiKey)
      .expect(200);

    expect(response.body).toHaveProperty('symbol', 'AAPL');
    expect(response.body).toHaveProperty('trend');
    expect(response.body).toHaveProperty('volatility');
  });

  test('POST /api/trade/execute requires authentication', async () => {
    await request(app)
      .post('/api/trade/execute')
      .send({
        strategy: 'momentum',
        symbol: 'AAPL',
        action: 'buy',
        quantity: 10
      })
      .expect(401);
  });
});
```

## Performance Optimization

### Connection Pooling

```javascript
// For database connections
const { Pool } = require('pg');

const pool = new Pool({
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
});

app.locals.db = pool;
```

### Clustering

```javascript
const cluster = require('cluster');
const numCPUs = require('os').cpus().length;

if (cluster.isMaster) {
  console.log(`Master ${process.pid} is running`);

  for (let i = 0; i < numCPUs; i++) {
    cluster.fork();
  }

  cluster.on('exit', (worker, code, signal) => {
    console.log(`Worker ${worker.process.pid} died`);
    cluster.fork();
  });
} else {
  initializeBackend().then(() => {
    app.listen(PORT);
    console.log(`Worker ${process.pid} started`);
  });
}
```

## Docker Deployment

```dockerfile
FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

EXPOSE 3000

CMD ["node", "server.js"]
```

## Troubleshooting

### Native Module Not Found

If you encounter "Cannot find module '*.node'":

1. Check platform compatibility: `node -p "process.platform + '-' + process.arch"`
2. Rebuild native modules: `npm rebuild @neural-trader/backend`
3. Clear npm cache: `npm cache clean --force`

### Performance Issues

1. Enable GPU acceleration where supported: `useGpu: true`
2. Implement caching for frequently accessed data
3. Use connection pooling for database access
4. Enable clustering for multi-core systems

### Memory Leaks

Monitor memory usage:

```javascript
setInterval(() => {
  const used = process.memoryUsage();
  console.log(`Memory usage: ${JSON.stringify(used, null, 2)}`);
}, 60000);
```

## Best Practices

1. **Always validate input** before passing to backend functions
2. **Use rate limiting** to prevent abuse
3. **Enable audit logging** for compliance and debugging
4. **Implement proper error handling** with meaningful error messages
5. **Use environment variables** for configuration
6. **Enable HTTPS** in production
7. **Monitor performance** and resource usage
8. **Implement graceful shutdown** for clean resource cleanup

## Additional Resources

- [Express.js Documentation](https://expressjs.com/)
- [NAPI-RS Documentation](https://napi.rs/)
- [Neural Trader GitHub](https://github.com/ruvnet/neural-trader)
