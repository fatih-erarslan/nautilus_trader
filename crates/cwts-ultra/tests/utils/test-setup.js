// Global test setup and utilities
const { performance } = require('perf_hooks');

// Global test configuration
global.TEST_TIMEOUT = 30000;
global.STRESS_TEST_TIMEOUT = 300000;
global.PERFORMANCE_THRESHOLD_MS = 100;
global.MEMORY_LEAK_THRESHOLD_MB = 50;

// Enhanced matchers for financial testing
expect.extend({
  toBeWithinFinancialTolerance(received, expected, tolerance = 0.0001) {
    const pass = Math.abs(received - expected) <= tolerance;
    return {
      message: () =>
        `expected ${received} to be within ${tolerance} of ${expected}`,
      pass,
    };
  },

  toBeValidPrice(received) {
    const pass = typeof received === 'number' && 
                 received >= 0 && 
                 !isNaN(received) && 
                 isFinite(received) &&
                 Number.isInteger(received * 10000); // 4 decimal places max
    return {
      message: () => `expected ${received} to be a valid financial price`,
      pass,
    };
  },

  toBeValidQuantity(received) {
    const pass = typeof received === 'number' && 
                 received > 0 && 
                 !isNaN(received) && 
                 isFinite(received);
    return {
      message: () => `expected ${received} to be a valid quantity`,
      pass,
    };
  },

  toBeValidTimestamp(received) {
    const pass = typeof received === 'number' && 
                 received > 0 && 
                 received <= Date.now() + 1000; // Allow 1 second future
    return {
      message: () => `expected ${received} to be a valid timestamp`,
      pass,
    };
  }
});

// Performance monitoring utilities
global.measurePerformance = (fn) => {
  return async (...args) => {
    const start = performance.now();
    const result = await fn(...args);
    const duration = performance.now() - start;
    return { result, duration };
  };
};

// Memory monitoring utilities
global.measureMemory = (fn) => {
  return async (...args) => {
    const startMemory = process.memoryUsage();
    const result = await fn(...args);
    if (global.gc) global.gc(); // Force garbage collection if available
    const endMemory = process.memoryUsage();
    
    return {
      result,
      memoryDelta: {
        heapUsed: endMemory.heapUsed - startMemory.heapUsed,
        heapTotal: endMemory.heapTotal - startMemory.heapTotal,
        external: endMemory.external - startMemory.external
      }
    };
  };
};

// Market data generators for testing
global.generateMarketData = {
  normalTick: () => ({
    symbol: 'AAPL',
    price: 150.00 + (Math.random() - 0.5) * 2,
    quantity: Math.floor(Math.random() * 1000) + 100,
    timestamp: Date.now(),
    side: Math.random() > 0.5 ? 'buy' : 'sell'
  }),

  extremeVolatility: () => ({
    symbol: 'AAPL',
    price: 150.00 * (0.1 + Math.random() * 1.8), // -90% to +80%
    quantity: Math.floor(Math.random() * 100000) + 1,
    timestamp: Date.now(),
    side: Math.random() > 0.5 ? 'buy' : 'sell'
  }),

  flashCrash: (basePrice = 150.00) => ({
    symbol: 'AAPL',
    price: basePrice * (0.01 + Math.random() * 0.09), // 99% drop
    quantity: Math.floor(Math.random() * 1000000) + 10000,
    timestamp: Date.now(),
    side: 'sell'
  })
};

// Async error handling
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  process.exit(1);
});

// Test isolation utilities
beforeEach(() => {
  jest.clearAllMocks();
  jest.resetModules();
});

afterEach(() => {
  // Cleanup any test artifacts
  if (global.gc) global.gc();
});