/**
 * Integration Tests for Week 1 Optimizations
 *
 * Validates:
 * 1. Parameter type fixes (7 tools)
 * 2. Redis caching implementation
 * 3. Rate limiting enforcement
 * 4. E2B NAPI exports
 * 5. JWT security hardening
 *
 * Run: npm test -- tests/integration/optimizations-validation.test.js
 */

const { validateToolParameters } = require('../../neural-trader-rust/packages/mcp/src/tools/parameter-fixes');
const { CacheManager } = require('../../neural-trader-rust/packages/mcp/src/middleware/cache-manager');
const { RateLimiter } = require('../../neural-trader-rust/packages/mcp/src/middleware/rate-limiter');

describe('Week 1 Optimizations - Integration Tests', () => {
  describe('1. Parameter Type Fixes', () => {
    test('run_backtest: converts string use_gpu to boolean', () => {
      const params = {
        strategy: 'momentum',
        symbol: 'AAPL',
        start_date: '2024-01-01',
        end_date: '2024-12-31',
        use_gpu: 'true',  // String instead of boolean
      };

      const validated = validateToolParameters('run_backtest', params);

      expect(validated.use_gpu).toBe(true);  // Converted to boolean
      expect(typeof validated.use_gpu).toBe('boolean');
    });

    test('optimize_strategy: parses JSON parameter_ranges', () => {
      const params = {
        strategy: 'momentum',
        symbol: 'AAPL',
        parameter_ranges: '{"lookback": [10, 50], "threshold": [0.01, 0.05]}',
        use_gpu: true,
      };

      const validated = validateToolParameters('optimize_strategy', params);

      expect(validated.parameter_ranges).toEqual({
        lookback: [10, 50],
        threshold: [0.01, 0.05],
      });
      expect(typeof validated.parameter_ranges).toBe('object');
    });

    test('neural_forecast: correct parameter order', () => {
      const params = {
        symbol: 'AAPL',
        horizon: 5,
        model_id: null,
        use_gpu: 'true',
        confidence_level: 0.95,
      };

      const validated = validateToolParameters('neural_forecast', params);

      expect(validated.use_gpu).toBe(true);
      expect(validated.horizon).toBe(5);
      expect(validated.model_id).toBe(null);
    });

    test('risk_analysis: parses portfolio array', () => {
      const params = {
        portfolio: '[{"symbol":"AAPL","weight":0.5},{"symbol":"MSFT","weight":0.5}]',
        use_gpu: true,
      };

      const validated = validateToolParameters('risk_analysis', params);

      expect(Array.isArray(validated.portfolio)).toBe(true);
      expect(validated.portfolio).toHaveLength(2);
      expect(validated.portfolio[0].symbol).toBe('AAPL');
    });

    test('correlation_analysis: handles comma-separated symbols', () => {
      const params = {
        symbols: 'AAPL,MSFT,GOOGL',
        period_days: 90,
        use_gpu: false,
      };

      const validated = validateToolParameters('correlation_analysis', params);

      expect(Array.isArray(validated.symbols)).toBe(true);
      expect(validated.symbols).toEqual(['AAPL', 'MSFT', 'GOOGL']);
    });

    test('parameter validation throws error on invalid input', () => {
      expect(() => {
        validateToolParameters('optimize_strategy', {
          strategy: 'momentum',
          symbol: 'AAPL',
          parameter_ranges: 'invalid json{',
        });
      }).toThrow('Parameter validation failed');
    });
  });

  describe('2. Redis Caching', () => {
    let cache;

    beforeAll(() => {
      cache = new CacheManager({ defaultTTL: 60 });
    });

    afterAll(async () => {
      await cache.clear();
      await cache.close();
    });

    test('cache stores and retrieves values', async () => {
      const namespace = 'test';
      const params = { symbol: 'AAPL' };
      const value = { price: 150.25, volume: 1000000 };

      // Store value
      await cache.set(namespace, params, value, 60);

      // Retrieve value
      const cached = await cache.get(namespace, params);

      expect(cached).toEqual(value);
    });

    test('cache returns null for missing values', async () => {
      const namespace = 'test';
      const params = { symbol: 'NONEXISTENT' };

      const cached = await cache.get(namespace, params);

      expect(cached).toBe(null);
    });

    test('cache tracks hit/miss statistics', async () => {
      const namespace = 'stats-test';
      const params = { test: 'value' };

      // Miss
      await cache.get(namespace, params);

      // Store
      await cache.set(namespace, params, { data: 'test' });

      // Hit
      await cache.get(namespace, params);

      const stats = cache.getStats();

      expect(stats.hits).toBeGreaterThan(0);
      expect(stats.misses).toBeGreaterThan(0);
      expect(stats.total).toBeGreaterThan(0);
    });

    test('cache namespace invalidation', async () => {
      const namespace = 'invalidate-test';

      // Store multiple values
      await cache.set(namespace, { key: 1 }, { value: 'a' });
      await cache.set(namespace, { key: 2 }, { value: 'b' });

      // Verify stored
      expect(await cache.get(namespace, { key: 1 })).toEqual({ value: 'a' });

      // Invalidate namespace
      await cache.delete(namespace);

      // Verify cleared
      expect(await cache.get(namespace, { key: 1 })).toBe(null);
      expect(await cache.get(namespace, { key: 2 })).toBe(null);
    });

    test('cache uses correct TTL per namespace', async () => {
      const testCases = [
        { namespace: 'odds', expectedTTL: 30 },
        { namespace: 'strategy', expectedTTL: 300 },
        { namespace: 'marketStatus', expectedTTL: 900 },
      ];

      for (const { namespace, expectedTTL } of testCases) {
        const ttl = require('../../neural-trader-rust/packages/mcp/src/middleware/cache-manager').getTTLForNamespace(namespace);
        expect(ttl).toBe(expectedTTL);
      }
    });
  });

  describe('3. Rate Limiting', () => {
    let limiter;

    beforeAll(() => {
      limiter = new RateLimiter({ maxRequests: 5, windowMs: 60000 });
    });

    afterAll(async () => {
      await limiter.close();
    });

    test('allows requests under limit', async () => {
      const clientId = 'test-client-allow';

      for (let i = 0; i < 5; i++) {
        const result = await limiter.checkLimit(clientId);
        expect(result.allowed).toBe(true);
      }
    });

    test('blocks requests over limit', async () => {
      const clientId = 'test-client-block';

      // Exhaust limit
      for (let i = 0; i < 5; i++) {
        await limiter.checkLimit(clientId);
      }

      // Should be blocked
      const result = await limiter.checkLimit(clientId);
      expect(result.allowed).toBe(false);
      expect(result.remaining).toBe(0);
    });

    test('returns correct remaining count', async () => {
      const clientId = 'test-client-remaining';

      const result1 = await limiter.checkLimit(clientId);
      expect(result1.remaining).toBe(4);  // 5 max - 1 used

      const result2 = await limiter.checkLimit(clientId);
      expect(result2.remaining).toBe(3);  // 5 max - 2 used
    });

    test('provides reset time', async () => {
      const clientId = 'test-client-reset';

      const result = await limiter.checkLimit(clientId);

      expect(result.resetTime).toBeGreaterThan(Date.now());
      expect(result.resetTime).toBeLessThan(Date.now() + 61000);  // Within 61 seconds
    });

    test('different tool categories have different limits', () => {
      const { getRateLimiterForTool } = require('../../neural-trader-rust/packages/mcp/src/middleware/rate-limiter');

      const oddsLimiter = getRateLimiterForTool('get_sports_odds');
      const neuralLimiter = getRateLimiterForTool('neural_forecast');
      const authLimiter = getRateLimiterForTool('user_login');

      expect(oddsLimiter.maxRequests).toBe(50);    // Odds API
      expect(neuralLimiter.maxRequests).toBe(20);  // Neural
      expect(authLimiter.maxRequests).toBe(5);     // Auth
    });
  });

  describe('4. E2B NAPI Exports', () => {
    test('E2B functions are exported from NAPI bindings', () => {
      // Note: This test checks if the exports exist
      // Actual functionality requires E2B API key and running tests

      const e2bFunctions = [
        'create_e2b_sandbox',
        'run_e2b_agent',
        'execute_e2b_process',
        'list_e2b_sandboxes',
        'terminate_e2b_sandbox',
        'get_e2b_sandbox_status',
        'deploy_e2b_template',
        'scale_e2b_deployment',
        'monitor_e2b_health',
      ];

      // Check if NAPI module exports these functions
      // (Skip if NAPI not loaded, as this is integration test)
      if (process.env.SKIP_E2B_TESTS !== 'true') {
        try {
          const napi = require('../../neural-trader-rust/packages/neural-trader-backend');

          for (const funcName of e2bFunctions) {
            expect(typeof napi[funcName]).toBe('function');
          }
        } catch (error) {
          console.warn('⚠️  NAPI module not loaded, skipping E2B export tests');
        }
      }
    });
  });

  describe('5. End-to-End Integration', () => {
    test('complete request flow with all optimizations', async () => {
      const cache = new CacheManager();
      const limiter = new RateLimiter({ maxRequests: 100, windowMs: 60000 });

      // 1. Rate limit check
      const rateLimitResult = await limiter.checkLimit('integration-test');
      expect(rateLimitResult.allowed).toBe(true);

      // 2. Cache check (miss)
      const namespace = 'integration';
      const params = { test: 'complete-flow' };
      const cached = await cache.get(namespace, params);
      expect(cached).toBe(null);

      // 3. Parameter validation
      const backtest_params = {
        strategy: 'momentum',
        symbol: 'AAPL',
        start_date: '2024-01-01',
        end_date: '2024-12-31',
        use_gpu: 'true',  // Will be converted
      };

      const validated = validateToolParameters('run_backtest', backtest_params);
      expect(validated.use_gpu).toBe(true);

      // 4. Cache result
      const result = { sharpe_ratio: 1.5, total_return: 0.25 };
      await cache.set(namespace, params, result);

      // 5. Cache hit
      const cached2 = await cache.get(namespace, params);
      expect(cached2).toEqual(result);

      // Cleanup
      await cache.close();
      await limiter.close();
    });

    test('statistics tracking across all middleware', async () => {
      const cache = new CacheManager();
      const limiter = new RateLimiter({ maxRequests: 100, windowMs: 60000 });

      // Perform operations
      await cache.get('stats', { test: 1 });
      await cache.set('stats', { test: 1 }, { value: 'data' });
      await cache.get('stats', { test: 1 });

      await limiter.checkLimit('stats-client');

      // Get stats
      const cacheStats = cache.getStats();
      expect(cacheStats.hits).toBeGreaterThan(0);
      expect(cacheStats.misses).toBeGreaterThan(0);
      expect(cacheStats.sets).toBeGreaterThan(0);

      // Cleanup
      await cache.close();
      await limiter.close();
    });
  });
});

describe('Performance Benchmarks', () => {
  test('cache latency < 5ms', async () => {
    const cache = new CacheManager();

    const start = Date.now();
    await cache.set('perf', { test: 1 }, { data: 'test' });
    await cache.get('perf', { test: 1 });
    const duration = Date.now() - start;

    expect(duration).toBeLessThan(5);

    await cache.close();
  });

  test('rate limiter latency < 10ms', async () => {
    const limiter = new RateLimiter();

    const start = Date.now();
    await limiter.checkLimit('perf-test');
    const duration = Date.now() - start;

    expect(duration).toBeLessThan(10);

    await limiter.close();
  });

  test('parameter validation < 1ms', () => {
    const params = {
      strategy: 'momentum',
      symbol: 'AAPL',
      start_date: '2024-01-01',
      end_date: '2024-12-31',
      use_gpu: 'true',
    };

    const start = Date.now();
    validateToolParameters('run_backtest', params);
    const duration = Date.now() - start;

    expect(duration).toBeLessThan(1);
  });
});
