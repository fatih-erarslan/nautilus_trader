/**
 * Setup file for CLI tests
 */

// Set test environment
process.env.NODE_ENV = 'test';

// Suppress console output during tests (optional)
if (process.env.SILENT_TESTS === 'true') {
  global.console = {
    ...console,
    log: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    info: jest.fn()
  };
}

// Global test timeout
jest.setTimeout(30000);

// Mock NAPI bindings for tests that don't need them
jest.mock('../../../index.js', () => ({
  fetchMarketData: jest.fn(),
  runStrategy: jest.fn(),
  backtest: jest.fn(),
  trainModel: jest.fn(),
  predict: jest.fn()
}), { virtual: true });

// Clean up after each test
afterEach(() => {
  jest.clearAllMocks();
});

// Global teardown
afterAll(() => {
  // Clean up any remaining temp files
  const fs = require('fs');
  const path = require('path');
  const os = require('os');

  try {
    const tempDir = os.tmpdir();
    const files = fs.readdirSync(tempDir);

    files.forEach(file => {
      if (file.startsWith('neural-trader-test-') || file.startsWith('neural-trader-e2e-')) {
        const fullPath = path.join(tempDir, file);
        try {
          if (fs.existsSync(fullPath)) {
            fs.rmSync(fullPath, { recursive: true, force: true });
          }
        } catch (err) {
          // Ignore cleanup errors
        }
      }
    });
  } catch (err) {
    // Ignore cleanup errors
  }
});
