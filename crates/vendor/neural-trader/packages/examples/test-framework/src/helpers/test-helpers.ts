/**
 * Generic test helper utilities
 */

import { TestConfig, TestMetrics } from '../types';

/**
 * Setup test environment with common configuration
 */
export function setupTestEnvironment(config: TestConfig = {}): void {
  const {
    timeout = 30000,
    verbose = false,
    mockOpenRouter = true,
    mockAgentDB = true
  } = config;

  // Set environment variables for testing
  process.env.NODE_ENV = 'test';
  process.env.LOG_LEVEL = verbose ? 'debug' : 'error';

  if (mockOpenRouter) {
    process.env.OPENROUTER_API_KEY = 'test-key';
  }

  if (mockAgentDB) {
    process.env.AGENTDB_MEMORY_ONLY = 'true';
  }

  // Set default timeout
  jest.setTimeout(timeout);
}

/**
 * Cleanup test environment
 */
export function cleanupTestEnvironment(): void {
  delete process.env.OPENROUTER_API_KEY;
  delete process.env.AGENTDB_MEMORY_ONLY;
}

/**
 * Generate random test data
 */
export function generateRandomData(count: number, min = 0, max = 100): number[] {
  return Array.from({ length: count }, () =>
    Math.random() * (max - min) + min
  );
}

/**
 * Calculate test metrics from predictions and actuals
 */
export function calculateMetrics(
  predictions: number[],
  actuals: number[],
  threshold = 0.5
): TestMetrics {
  if (predictions.length !== actuals.length) {
    throw new Error('Predictions and actuals must have same length');
  }

  let tp = 0, fp = 0, tn = 0, fn = 0;

  for (let i = 0; i < predictions.length; i++) {
    const pred = predictions[i] >= threshold ? 1 : 0;
    const actual = actuals[i] >= threshold ? 1 : 0;

    if (pred === 1 && actual === 1) tp++;
    else if (pred === 1 && actual === 0) fp++;
    else if (pred === 0 && actual === 0) tn++;
    else fn++;
  }

  const accuracy = (tp + tn) / predictions.length;
  const precision = tp / (tp + fp) || 0;
  const recall = tp / (tp + fn) || 0;
  const f1Score = 2 * (precision * recall) / (precision + recall) || 0;

  return {
    accuracy,
    precision,
    recall,
    f1Score,
    predictions: predictions.length,
    errors: fp + fn
  };
}

/**
 * Assert that a value is within a percentage of expected
 */
export function assertWithinPercent(
  actual: number,
  expected: number,
  percentTolerance: number
): boolean {
  const tolerance = Math.abs(expected * percentTolerance / 100);
  const diff = Math.abs(actual - expected);
  return diff <= tolerance;
}

/**
 * Wait for a condition to be true
 */
export async function waitForCondition(
  condition: () => boolean | Promise<boolean>,
  timeout = 5000,
  interval = 100
): Promise<void> {
  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    if (await condition()) {
      return;
    }
    await new Promise(resolve => setTimeout(resolve, interval));
  }

  throw new Error(`Condition not met within ${timeout}ms`);
}

/**
 * Retry a function with exponential backoff
 */
export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries = 3,
  initialDelay = 100
): Promise<T> {
  let lastError: Error;

  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;
      if (i < maxRetries - 1) {
        await new Promise(resolve =>
          setTimeout(resolve, initialDelay * Math.pow(2, i))
        );
      }
    }
  }

  throw lastError!;
}

/**
 * Measure execution time of a function
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
 * Create a mock spy that tracks calls
 */
export function createSpy<T extends (...args: any[]) => any>(): jest.Mock<T> {
  return jest.fn();
}
