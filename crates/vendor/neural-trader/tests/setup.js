/**
 * Jest setup file for E2B benchmarks
 */

// Extend Jest timeout for long-running benchmarks
jest.setTimeout(600000); // 10 minutes

// Setup environment
process.env.NODE_ENV = 'test';

// Check for E2B API key
if (!process.env.E2B_API_KEY) {
  console.warn('\n⚠️  WARNING: E2B_API_KEY not set. E2B tests will be skipped.\n');
  console.warn('To run E2B benchmarks, set your API key:');
  console.warn('  export E2B_API_KEY="your-api-key-here"\n');
}

// Global setup
beforeAll(() => {
  console.log('\n🚀 Starting benchmark suite...\n');
});

// Global teardown
afterAll(() => {
  console.log('\n✅ Benchmark suite completed!\n');
});

// Error handling
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

module.exports = {};
