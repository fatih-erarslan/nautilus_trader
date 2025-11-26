/**
 * Custom Jest matchers for neural-trader testing
 */

declare global {
  namespace jest {
    interface Matchers<R> {
      toBeWithinPercent(expected: number, percent: number): R;
      toBeValidPrediction(): R;
      toBeValidTimeSeries(): R;
      toHaveConverged(threshold?: number): R;
      toHaveImproved(baseline: number): R;
    }
  }
}

/**
 * Check if value is within percentage of expected
 */
export const toBeWithinPercent: jest.CustomMatcher = function (
  received: number,
  expected: number,
  percent: number
) {
  const tolerance = Math.abs(expected * percent / 100);
  const diff = Math.abs(received - expected);
  const pass = diff <= tolerance;

  return {
    pass,
    message: () =>
      pass
        ? `Expected ${received} not to be within ${percent}% of ${expected}`
        : `Expected ${received} to be within ${percent}% of ${expected} (tolerance: ${tolerance}, diff: ${diff})`
  };
};

/**
 * Check if value is a valid prediction
 */
export const toBeValidPrediction: jest.CustomMatcher = function (
  received: any
) {
  const isValid =
    typeof received === 'number' &&
    !isNaN(received) &&
    isFinite(received) &&
    received >= 0 &&
    received <= 1;

  return {
    pass: isValid,
    message: () =>
      isValid
        ? `Expected ${received} not to be a valid prediction`
        : `Expected ${received} to be a valid prediction (number between 0 and 1)`
  };
};

/**
 * Check if array is a valid time series
 */
export const toBeValidTimeSeries: jest.CustomMatcher = function (
  received: any
) {
  const isValid =
    Array.isArray(received) &&
    received.length > 0 &&
    received.every(v => typeof v === 'number' && !isNaN(v));

  return {
    pass: isValid,
    message: () =>
      isValid
        ? `Expected ${received} not to be a valid time series`
        : `Expected ${received} to be a valid time series (non-empty array of numbers)`
  };
};

/**
 * Check if values have converged
 */
export const toHaveConverged: jest.CustomMatcher = function (
  received: number[],
  threshold = 0.01
) {
  if (!Array.isArray(received) || received.length < 2) {
    return {
      pass: false,
      message: () => 'Expected array with at least 2 values'
    };
  }

  const last10 = received.slice(-10);
  const mean = last10.reduce((a, b) => a + b, 0) / last10.length;
  const variance = last10.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / last10.length;
  const stdDev = Math.sqrt(variance);

  const hasConverged = stdDev / Math.abs(mean) < threshold;

  return {
    pass: hasConverged,
    message: () =>
      hasConverged
        ? `Expected values not to have converged (stdDev: ${stdDev}, mean: ${mean})`
        : `Expected values to have converged (stdDev: ${stdDev}, mean: ${mean}, threshold: ${threshold})`
  };
};

/**
 * Check if value has improved over baseline
 */
export const toHaveImproved: jest.CustomMatcher = function (
  received: number,
  baseline: number
) {
  const pass = received > baseline;

  return {
    pass,
    message: () =>
      pass
        ? `Expected ${received} not to have improved over ${baseline}`
        : `Expected ${received} to have improved over ${baseline}`
  };
};

/**
 * Install custom matchers
 */
export function installMatchers(): void {
  expect.extend({
    toBeWithinPercent,
    toBeValidPrediction,
    toBeValidTimeSeries,
    toHaveConverged,
    toHaveImproved
  });
}
