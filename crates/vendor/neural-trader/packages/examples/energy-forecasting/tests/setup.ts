/**
 * Jest test setup
 */

// Increase test timeout for model training
jest.setTimeout(30000);

// Mock console methods to reduce noise
global.console = {
  ...console,
  info: jest.fn(),
  warn: jest.fn(),
  // Keep error for debugging
  error: console.error
};
