/**
 * Express.js Integration Example
 *
 * Complete production-ready Express server with neural-trader backend
 * Features:
 * - Authentication & authorization
 * - Rate limiting & DDoS protection
 * - Audit logging
 * - CORS & security headers
 * - Input sanitization
 * - Health checks
 * - Graceful shutdown
 */

const express = require('express');
const backend = require('@neural-trader/backend');
const dotenv = require('dotenv');

// Load environment variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Initialize backend
let initialized = false;

async function initializeBackend() {
  if (initialized) return;

  try {
    const jwtSecret = process.env.JWT_SECRET;
    if (!jwtSecret) {
      throw new Error('JWT_SECRET environment variable is required');
    }

    // Initialize authentication
    backend.initAuth(jwtSecret);
    console.log('✓ Authentication initialized');

    // Initialize rate limiter
    backend.initRateLimiter({
      maxRequestsPerMinute: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS) || 100,
      burstSize: parseInt(process.env.RATE_LIMIT_BURST_SIZE) || 20,
      windowDurationSecs: parseInt(process.env.RATE_LIMIT_WINDOW_SECS) || 60
    });
    console.log('✓ Rate limiter initialized');

    // Initialize audit logger
    backend.initAuditLogger(
      parseInt(process.env.AUDIT_MAX_EVENTS) || 10000,
      process.env.AUDIT_LOG_CONSOLE !== 'false',
      process.env.AUDIT_LOG_FILE !== 'false'
    );
    console.log('✓ Audit logger initialized');

    // Initialize security config
    const corsOrigins = (process.env.CORS_ALLOWED_ORIGINS || '*').split(',');
    backend.initSecurityConfig(
      {
        allowedOrigins: corsOrigins,
        allowedMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
        allowedHeaders: ['Content-Type', 'Authorization', 'X-API-Key'],
        exposedHeaders: ['X-Total-Count', 'X-Page-Number'],
        allowCredentials: true,
        maxAge: 86400
      },
      process.env.REQUIRE_HTTPS !== 'false'
    );
    console.log('✓ Security config initialized');

    // Initialize neural trader
    await backend.initNeuralTrader(JSON.stringify({
      logLevel: process.env.LOG_LEVEL || 'info',
      enableGpu: process.env.ENABLE_GPU === 'true'
    }));
    console.log('✓ Neural Trader initialized');

    initialized = true;

    const systemInfo = backend.getSystemInfo();
    console.log(`\n🚀 Neural Trader v${systemInfo.version} ready`);
    console.log(`   Features: ${systemInfo.features.join(', ')}`);
    console.log(`   Total tools: ${systemInfo.totalTools}\n`);
  } catch (error) {
    console.error('❌ Failed to initialize backend:', error.message);
    throw error;
  }
}

// ========================================
// Middleware
// ========================================

// CORS middleware
app.use((req, res, next) => {
  const origin = req.headers.origin;

  if (backend.checkCorsOrigin(origin || '*')) {
    const corsHeaders = JSON.parse(backend.getCorsHeaders(origin));

    for (const [key, value] of Object.entries(corsHeaders)) {
      res.setHeader(key, value);
    }
  }

  if (req.method === 'OPTIONS') {
    return res.sendStatus(200);
  }

  next();
});

// Security headers middleware
app.use((req, res, next) => {
  const headers = JSON.parse(backend.getSecurityHeaders());

  for (const [key, value] of Object.entries(headers)) {
    res.setHeader(key, value);
  }

  next();
});

// Authentication middleware
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
      req.ip,
      req.path
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
      req.ip,
      req.path
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
      req.ip,
      req.path,
      JSON.stringify({ error: error.message })
    );
    res.status(401).json({ error: 'Invalid API key' });
  }
};

// Rate limiting middleware
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
      req.ip,
      req.path
    );

    const stats = backend.getRateLimitStats(identifier);
    return res.status(429).json({
      error: 'Rate limit exceeded',
      retryAfter: Math.ceil((60 - (stats.totalRequests % 60)) / stats.refillRate),
      stats: {
        tokensAvailable: stats.tokensAvailable,
        maxTokens: stats.maxTokens,
        successRate: stats.successRate
      }
    });
  }

  next();
};

// Input sanitization middleware
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
            req.path,
            JSON.stringify({ key, threats, value: sanitized.substring(0, 100) })
          );

          return res.status(400).json({
            error: 'Security threat detected in input',
            field: key,
            threats
          });
        }

        req.body[key] = sanitized;
      }
    }
  }

  next();
};

// Request logging middleware
app.use((req, res, next) => {
  const startTime = Date.now();

  res.on('finish', () => {
    const duration = Date.now() - startTime;

    backend.logAuditEvent(
      res.statusCode >= 400 ? 'Warning' : 'Info',
      'DataAccess',
      `${req.method} ${req.path}`,
      res.statusCode < 400 ? 'success' : 'failure',
      req.user?.userId,
      req.user?.username,
      req.ip,
      req.path,
      JSON.stringify({
        duration,
        statusCode: res.statusCode,
        contentLength: res.get('Content-Length')
      })
    );
  });

  next();
});

// ========================================
// Public Routes
// ========================================

// Health check
app.get('/health', async (req, res) => {
  try {
    const health = await backend.healthCheck();
    res.json(health);
  } catch (error) {
    res.status(500).json({
      status: 'unhealthy',
      error: error.message
    });
  }
});

// System info
app.get('/api/system/info', (req, res) => {
  try {
    const info = backend.getSystemInfo();
    res.json(info);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ========================================
// Protected Routes - Trading
// ========================================

// List strategies
app.get('/api/strategies', authMiddleware, rateLimitMiddleware, async (req, res) => {
  try {
    const strategies = await backend.listStrategies();
    res.json({ strategies });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get strategy info
app.get('/api/strategies/:name', authMiddleware, rateLimitMiddleware, async (req, res) => {
  try {
    const info = await backend.getStrategyInfo(req.params.name);
    res.json({ strategy: req.params.name, info: JSON.parse(info) });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Quick market analysis
app.get('/api/analysis/:symbol', authMiddleware, rateLimitMiddleware, async (req, res) => {
  try {
    const { symbol } = req.params;
    const { useGpu = 'false' } = req.query;

    const analysis = await backend.quickAnalysis(symbol, useGpu === 'true');

    res.json({
      symbol,
      analysis,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Simulate trade
app.post('/api/trade/simulate',
  authMiddleware,
  rateLimitMiddleware,
  sanitizeMiddleware,
  async (req, res) => {
    try {
      const { strategy, symbol, action, useGpu = false } = req.body;

      if (!strategy || !symbol || !action) {
        return res.status(400).json({
          error: 'Missing required fields: strategy, symbol, action'
        });
      }

      const result = await backend.simulateTrade(strategy, symbol, action, useGpu);

      res.json({
        simulation: result,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }
);

// Execute trade
app.post('/api/trade/execute',
  authMiddleware,
  rateLimitMiddleware,
  sanitizeMiddleware,
  async (req, res) => {
    try {
      const {
        strategy,
        symbol,
        action,
        quantity,
        orderType = 'market',
        limitPrice
      } = req.body;

      // Validate required fields
      if (!strategy || !symbol || !action || !quantity) {
        return res.status(400).json({
          error: 'Missing required fields: strategy, symbol, action, quantity'
        });
      }

      // Validate trading parameters
      if (!backend.validateTradingParams(symbol, quantity, limitPrice)) {
        return res.status(400).json({
          error: 'Invalid trading parameters'
        });
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
          'unauthorized_trade_attempt',
          'failure',
          req.user.userId,
          req.user.username,
          req.ip,
          symbol,
          JSON.stringify({ strategy, action, quantity })
        );

        return res.status(403).json({
          error: 'Insufficient permissions to execute trades'
        });
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
        JSON.stringify({
          orderId: result.orderId,
          strategy,
          action,
          quantity,
          orderType
        })
      );

      res.json({
        trade: result,
        timestamp: new Date().toISOString()
      });
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

// ========================================
// Protected Routes - Portfolio
// ========================================

// Get portfolio status
app.get('/api/portfolio', authMiddleware, rateLimitMiddleware, async (req, res) => {
  try {
    const { includeAnalytics = 'true' } = req.query;

    const status = await backend.getPortfolioStatus(includeAnalytics === 'true');

    res.json({
      portfolio: status,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Portfolio risk analysis
app.post('/api/portfolio/risk',
  authMiddleware,
  rateLimitMiddleware,
  async (req, res) => {
    try {
      const { portfolio, useGpu = false } = req.body;

      if (!portfolio) {
        return res.status(400).json({
          error: 'Portfolio data required'
        });
      }

      const analysis = await backend.riskAnalysis(
        JSON.stringify(portfolio),
        useGpu
      );

      res.json({
        riskAnalysis: analysis,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }
);

// ========================================
// Protected Routes - Backtesting
// ========================================

// Run backtest
app.post('/api/backtest',
  authMiddleware,
  rateLimitMiddleware,
  sanitizeMiddleware,
  async (req, res) => {
    try {
      const {
        strategy,
        symbol,
        startDate,
        endDate,
        useGpu = false
      } = req.body;

      if (!strategy || !symbol || !startDate || !endDate) {
        return res.status(400).json({
          error: 'Missing required fields: strategy, symbol, startDate, endDate'
        });
      }

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

      res.json({
        backtest: result,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }
);

// ========================================
// Protected Routes - E2B Swarm
// ========================================

// Initialize swarm
app.post('/api/swarm/init',
  authMiddleware,
  rateLimitMiddleware,
  async (req, res) => {
    try {
      const { topology, config } = req.body;

      if (!topology || !config) {
        return res.status(400).json({
          error: 'Missing required fields: topology, config'
        });
      }

      const swarm = await backend.initE2bSwarm(
        topology,
        JSON.stringify(config)
      );

      res.json({
        swarm,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }
);

// Get swarm status
app.get('/api/swarm/:swarmId/status',
  authMiddleware,
  rateLimitMiddleware,
  async (req, res) => {
    try {
      const { swarmId } = req.params;

      const status = await backend.getSwarmStatus(swarmId);

      res.json({
        swarm: status,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }
);

// ========================================
// Admin Routes
// ========================================

// Create API key (admin only)
app.post('/api/admin/api-keys',
  authMiddleware,
  rateLimitMiddleware,
  sanitizeMiddleware,
  (req, res) => {
    try {
      // Check admin permission
      if (req.user.role !== 'Admin') {
        return res.status(403).json({ error: 'Admin access required' });
      }

      const { username, role, rateLimit, expiresInDays } = req.body;

      if (!username || !role) {
        return res.status(400).json({
          error: 'Missing required fields: username, role'
        });
      }

      const apiKey = backend.createApiKey(
        username,
        role,
        rateLimit,
        expiresInDays
      );

      backend.logAuditEvent(
        'Info',
        'Configuration',
        'api_key_created',
        'success',
        req.user.userId,
        req.user.username,
        req.ip,
        username,
        JSON.stringify({ role, rateLimit, expiresInDays })
      );

      res.json({
        apiKey,
        message: 'API key created successfully. Store this securely - it cannot be retrieved again.'
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }
);

// Get audit events
app.get('/api/admin/audit',
  authMiddleware,
  rateLimitMiddleware,
  (req, res) => {
    try {
      if (req.user.role !== 'Admin') {
        return res.status(403).json({ error: 'Admin access required' });
      }

      const { limit = '100' } = req.query;

      const events = backend.getAuditEvents(parseInt(limit));

      res.json({
        events,
        count: events.length,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }
);

// ========================================
// Error Handling
// ========================================

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Endpoint not found',
    path: req.path,
    method: req.method
  });
});

// Global error handler
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);

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
    requestId: Date.now().toString()
  });
});

// ========================================
// Server Startup
// ========================================

async function startServer() {
  try {
    await initializeBackend();

    app.listen(PORT, () => {
      console.log(`\n✓ Server listening on port ${PORT}`);
      console.log(`  Health: http://localhost:${PORT}/health`);
      console.log(`  System Info: http://localhost:${PORT}/api/system/info\n`);
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('\nSIGTERM received, shutting down gracefully...');

  try {
    await backend.shutdown();
    console.log('✓ Backend shut down successfully');
    process.exit(0);
  } catch (error) {
    console.error('Error during shutdown:', error);
    process.exit(1);
  }
});

process.on('SIGINT', async () => {
  console.log('\nSIGINT received, shutting down gracefully...');

  try {
    await backend.shutdown();
    console.log('✓ Backend shut down successfully');
    process.exit(0);
  } catch (error) {
    console.error('Error during shutdown:', error);
    process.exit(1);
  }
});

// Start the server
startServer();

module.exports = app;
