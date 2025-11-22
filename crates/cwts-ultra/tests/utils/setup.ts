/**
 * Global Test Setup - Configures the testing environment
 * Initializes all test utilities and framework components
 */

import { jest } from '@jest/globals';
import * as path from 'path';

// Global test configuration
global.beforeAll = global.beforeAll || (() => {});
global.afterAll = global.afterAll || (() => {});
global.beforeEach = global.beforeEach || (() => {});
global.afterEach = global.afterEach || (() => {});

// Set test environment variables
process.env.NODE_ENV = 'test';
process.env.RUST_TEST_THREADS = '1';
process.env.PYTHONPATH = path.join(__dirname, '../../freqtrade');
process.env.TEST_MODE = 'comprehensive';

// Configure console output for tests
const originalConsoleError = console.error;
const originalConsoleWarn = console.warn;

console.error = (...args: any[]) => {
  // Filter out expected test warnings/errors
  const message = args.join(' ');
  if (!message.includes('Warning: ReactDOM.render') && !message.includes('act() warning')) {
    originalConsoleError.apply(console, args);
  }
};

console.warn = (...args: any[]) => {
  const message = args.join(' ');
  if (!message.includes('deprecated') && !message.includes('experimental')) {
    originalConsoleWarn.apply(console, args);
  }
};

// Global test timeout
jest.setTimeout(30000);

// Mock external dependencies
jest.mock('fs', () => ({
  ...jest.requireActual('fs'),
  promises: {
    ...jest.requireActual('fs').promises,
    readFile: jest.fn(),
    writeFile: jest.fn(),
    mkdir: jest.fn()
  }
}));

// Mock network requests
global.fetch = jest.fn(() =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve({}),
    text: () => Promise.resolve(''),
    status: 200,
    statusText: 'OK'
  })
) as jest.Mock;

// Global test utilities
(global as any).testUtils = {
  // Test data generators
  generateMarketData: (count: number = 100) => {
    return Array.from({ length: count }, (_, i) => ({
      id: i,
      symbol: 'BTCUSD',
      price: 50000 + Math.random() * 1000 - 500,
      volume: Math.random() * 1000,
      timestamp: Date.now() + i * 1000
    }));
  },

  // Mock implementations
  createMockOrderbook: () => ({
    buy: [],
    sell: [],
    spread: 0.1,
    depth: 10
  }),

  createMockRiskParams: () => ({
    maxPositionSize: 1000000,
    maxDrawdown: 0.05,
    stopLoss: 0.02,
    takeProfit: 0.04,
    riskPerTrade: 0.01
  }),

  // Test assertions helpers
  expectWithinTolerance: (actual: number, expected: number, tolerance: number = 1e-10) => {
    expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tolerance);
  },

  expectArraysEqual: (actual: any[], expected: any[], tolerance?: number) => {
    expect(actual).toHaveLength(expected.length);
    for (let i = 0; i < actual.length; i++) {
      if (typeof actual[i] === 'number' && typeof expected[i] === 'number') {
        if (tolerance !== undefined) {
          expect(Math.abs(actual[i] - expected[i])).toBeLessThanOrEqual(tolerance);
        } else {
          expect(actual[i]).toBe(expected[i]);
        }
      } else {
        expect(actual[i]).toEqual(expected[i]);
      }
    }
  }
};

// Performance monitoring setup
const performanceMarks: Map<string, number> = new Map();

(global as any).performanceUtils = {
  startMark: (name: string) => {
    performanceMarks.set(name, performance.now());
  },

  endMark: (name: string): number => {
    const start = performanceMarks.get(name);
    if (start === undefined) {
      throw new Error(`No start mark found for: ${name}`);
    }
    const duration = performance.now() - start;
    performanceMarks.delete(name);
    return duration;
  },

  measurePerformance: async <T>(name: string, fn: () => Promise<T>): Promise<{ result: T; duration: number }> => {
    const start = performance.now();
    const result = await fn();
    const duration = performance.now() - start;
    return { result, duration };
  }
};

// Memory usage tracking
(global as any).memoryUtils = {
  getMemoryUsage: () => {
    if (typeof process !== 'undefined' && process.memoryUsage) {
      return process.memoryUsage();
    }
    return {
      rss: 0,
      heapTotal: 0,
      heapUsed: 0,
      external: 0,
      arrayBuffers: 0
    };
  },

  trackMemoryLeak: async <T>(fn: () => Promise<T>): Promise<{ result: T; memoryDelta: number }> => {
    // Force garbage collection if available
    if (global.gc) {
      global.gc();
    }
    
    const beforeMemory = (global as any).memoryUtils.getMemoryUsage();
    const result = await fn();
    
    if (global.gc) {
      global.gc();
    }
    
    const afterMemory = (global as any).memoryUtils.getMemoryUsage();
    const memoryDelta = afterMemory.heapUsed - beforeMemory.heapUsed;
    
    return { result, memoryDelta };
  }
};

// Async error handling
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  // Don't exit in test environment, just log
});

process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  // Don't exit in test environment, just log
});

// Cleanup function for test suites
(global as any).cleanupTestEnvironment = async () => {
  // Clear all timers
  jest.clearAllTimers();
  
  // Clear all mocks
  jest.clearAllMocks();
  
  // Reset modules
  jest.resetModules();
  
  // Force garbage collection if available
  if (global.gc) {
    global.gc();
  }
  
  console.log('✅ Test environment cleaned up');
};

console.log('🚀 Test environment initialized successfully');