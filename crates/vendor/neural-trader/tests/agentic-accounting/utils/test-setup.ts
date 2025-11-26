/**
 * Test Setup and Utilities
 *
 * Configures the test environment, custom matchers, and global utilities
 */

import { beforeAll, afterAll, beforeEach, afterEach } from '@jest/globals';
import Decimal from 'decimal.js';

// ============================================================================
// Global Test Configuration
// ============================================================================

beforeAll(() => {
  // Set timezone to UTC for consistent date testing
  process.env.TZ = 'UTC';

  // Configure Decimal.js for consistent precision
  Decimal.set({
    precision: 30,
    rounding: Decimal.ROUND_HALF_UP,
    toExpNeg: -9e15,
    toExpPos: 9e15,
  });

  console.log('🧪 Test environment initialized');
});

afterAll(() => {
  console.log('✅ Test suite completed');
});

// ============================================================================
// Custom Matchers
// ============================================================================

interface CustomMatchers<R = unknown> {
  toBeDecimal(expected: string | number): R;
  toBeDecimalCloseTo(expected: string | number, precision?: number): R;
  toBeWithinRange(min: number, max: number): R;
  toHaveTransactionType(type: string): R;
  toBeValidDate(): R;
  toBeValidUUID(): R;
}

declare global {
  namespace jest {
    interface Expect extends CustomMatchers {}
    interface Matchers<R> extends CustomMatchers<R> {}
    interface InverseAsymmetricMatchers extends CustomMatchers {}
  }
}

expect.extend({
  /**
   * Checks if value equals expected decimal value
   */
  toBeDecimal(received: any, expected: string | number) {
    const receivedDecimal = new Decimal(received);
    const expectedDecimal = new Decimal(expected);
    const pass = receivedDecimal.equals(expectedDecimal);

    return {
      pass,
      message: () =>
        pass
          ? `Expected ${received} not to equal decimal ${expected}`
          : `Expected ${received} to equal decimal ${expected}, but got ${receivedDecimal.toString()}`,
    };
  },

  /**
   * Checks if decimal value is close to expected (within precision)
   */
  toBeDecimalCloseTo(received: any, expected: string | number, precision: number = 8) {
    const receivedDecimal = new Decimal(received);
    const expectedDecimal = new Decimal(expected);
    const diff = receivedDecimal.minus(expectedDecimal).abs();
    const tolerance = new Decimal(10).pow(-precision);
    const pass = diff.lte(tolerance);

    return {
      pass,
      message: () =>
        pass
          ? `Expected ${received} not to be close to decimal ${expected}`
          : `Expected ${received} to be close to decimal ${expected} (precision: ${precision}), but difference was ${diff.toString()}`,
    };
  },

  /**
   * Checks if number is within range [min, max]
   */
  toBeWithinRange(received: number, min: number, max: number) {
    const pass = received >= min && received <= max;

    return {
      pass,
      message: () =>
        pass
          ? `Expected ${received} not to be within range [${min}, ${max}]`
          : `Expected ${received} to be within range [${min}, ${max}]`,
    };
  },

  /**
   * Checks if object has specific transaction type
   */
  toHaveTransactionType(received: any, type: string) {
    const pass = received && received.type === type;

    return {
      pass,
      message: () =>
        pass
          ? `Expected transaction not to have type ${type}`
          : `Expected transaction to have type ${type}, but got ${received?.type}`,
    };
  },

  /**
   * Checks if value is a valid date
   */
  toBeValidDate(received: any) {
    const date = new Date(received);
    const pass = !isNaN(date.getTime());

    return {
      pass,
      message: () =>
        pass
          ? `Expected ${received} not to be a valid date`
          : `Expected ${received} to be a valid date`,
    };
  },

  /**
   * Checks if string is a valid UUID
   */
  toBeValidUUID(received: any) {
    const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
    const pass = typeof received === 'string' && uuidRegex.test(received);

    return {
      pass,
      message: () =>
        pass
          ? `Expected ${received} not to be a valid UUID`
          : `Expected ${received} to be a valid UUID`,
    };
  },
});

// ============================================================================
// Global Test Utilities
// ============================================================================

/**
 * Waits for a condition to be true with timeout
 */
export async function waitFor(
  condition: () => boolean | Promise<boolean>,
  timeout: number = 5000,
  interval: number = 100
): Promise<void> {
  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    if (await condition()) {
      return;
    }
    await new Promise(resolve => setTimeout(resolve, interval));
  }

  throw new Error(`Timeout waiting for condition after ${timeout}ms`);
}

/**
 * Sleeps for specified milliseconds
 */
export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Captures console output during test
 */
export function captureConsole() {
  const logs: string[] = [];
  const errors: string[] = [];
  const warns: string[] = [];

  const originalLog = console.log;
  const originalError = console.error;
  const originalWarn = console.warn;

  console.log = (...args: any[]) => logs.push(args.join(' '));
  console.error = (...args: any[]) => errors.push(args.join(' '));
  console.warn = (...args: any[]) => warns.push(args.join(' '));

  return {
    logs,
    errors,
    warns,
    restore: () => {
      console.log = originalLog;
      console.error = originalError;
      console.warn = originalWarn;
    },
  };
}

/**
 * Measures execution time of async function
 */
export async function measureTime<T>(
  fn: () => Promise<T>
): Promise<{ result: T; duration: number }> {
  const start = performance.now();
  const result = await fn();
  const duration = performance.now() - start;

  return { result, duration };
}

/**
 * Creates a mock timer for testing time-dependent code
 */
export function createMockTimer() {
  let currentTime = Date.now();

  return {
    now: () => currentTime,
    advance: (ms: number) => {
      currentTime += ms;
    },
    reset: () => {
      currentTime = Date.now();
    },
  };
}

// ============================================================================
// Export test utilities
// ============================================================================

export const testUtils = {
  waitFor,
  sleep,
  captureConsole,
  measureTime,
  createMockTimer,
};
