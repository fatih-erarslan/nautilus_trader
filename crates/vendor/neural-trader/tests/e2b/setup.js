/**
 * Jest Setup for E2B Template Tests
 */

// Set test environment variables
process.env.NODE_ENV = 'test';
process.env.TEST_MODE = 'e2b-templates';

// Extend timeout for CI environments
if (process.env.CI) {
  jest.setTimeout(180000); // 3 minutes in CI
}

// Global test utilities
global.sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Log test start
console.log('🧪 E2B Template Tests - Setup Complete');
console.log('Environment:', process.env.NODE_ENV);
console.log('Timeout:', jest.getDefaultTimeout ? jest.getDefaultTimeout() : 'N/A');

// Cleanup on exit
process.on('exit', () => {
  console.log('🏁 E2B Template Tests - Exiting');
});
