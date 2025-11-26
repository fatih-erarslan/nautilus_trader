/**
 * Jest Setup File for Backend Tests
 * Runs before each test file
 */

// Set test environment variables
process.env.NODE_ENV = 'test';
process.env.LOG_LEVEL = 'error'; // Reduce logging noise in tests

// Global test timeout
jest.setTimeout(30000);

// Mock data generators
global.generateMockOpportunity = (overrides = {}) => ({
  sport: 'nfl',
  event: 'Team A vs Team B',
  betType: 'moneyline',
  selection: 'Team A',
  odds: 2.5,
  probability: 0.55,
  edge: 0.15,
  confidence: 0.85,
  modelAgreement: 0.90,
  timeUntilEventSecs: 7200,
  liquidity: 0.8,
  isLive: false,
  isParlay: false,
  ...overrides
});

global.generateMockMember = (overrides = {}) => ({
  memberId: `member_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
  name: 'Test Member',
  email: 'test@example.com',
  capitalContribution: '10000',
  performanceScore: 0.5,
  tier: 'bronze',
  ...overrides
});

global.generateMockSwarmConfig = (overrides = {}) => ({
  maxAgents: 5,
  distributionStrategy: 0,
  enableGpu: false,
  autoScaling: false,
  minAgents: 2,
  maxMemoryMb: 512,
  timeoutSecs: 300,
  ...overrides
});

// Cleanup helpers
global.cleanupSyndicates = [];
global.cleanupSwarms = [];

global.registerSyndicateForCleanup = (syndicateId) => {
  global.cleanupSyndicates.push(syndicateId);
};

global.registerSwarmForCleanup = (swarmId) => {
  global.cleanupSwarms.push(swarmId);
};

// Cleanup after all tests
afterAll(async () => {
  // Cleanup is handled by the backend automatically
  // This is just for tracking
  console.log(`\nTest cleanup: ${global.cleanupSyndicates.length} syndicates, ${global.cleanupSwarms.length} swarms`);
});

// Custom matchers
expect.extend({
  toBeWithinRange(received, floor, ceiling) {
    const pass = received >= floor && received <= ceiling;
    if (pass) {
      return {
        message: () => `expected ${received} not to be within range ${floor} - ${ceiling}`,
        pass: true
      };
    } else {
      return {
        message: () => `expected ${received} to be within range ${floor} - ${ceiling}`,
        pass: false
      };
    }
  },

  toBeValidJSON(received) {
    try {
      JSON.parse(received);
      return {
        message: () => `expected ${received} not to be valid JSON`,
        pass: true
      };
    } catch (e) {
      return {
        message: () => `expected ${received} to be valid JSON, but parsing failed: ${e.message}`,
        pass: false
      };
    }
  },

  toHaveValidStructure(received, expectedKeys) {
    const keys = Object.keys(received);
    const missingKeys = expectedKeys.filter(key => !keys.includes(key));

    if (missingKeys.length === 0) {
      return {
        message: () => `expected object not to have all keys: ${expectedKeys.join(', ')}`,
        pass: true
      };
    } else {
      return {
        message: () => `expected object to have keys: ${expectedKeys.join(', ')}, missing: ${missingKeys.join(', ')}`,
        pass: false
      };
    }
  }
});

// Console helpers for debugging
global.logTestSection = (title) => {
  console.log(`\n${'='.repeat(60)}`);
  console.log(` ${title}`);
  console.log(`${'='.repeat(60)}\n`);
};

// Performance measurement helper
global.measurePerformance = async (name, fn) => {
  const start = process.hrtime.bigint();
  const result = await fn();
  const end = process.hrtime.bigint();
  const duration = Number(end - start) / 1_000_000; // Convert to ms

  console.log(`⏱️  ${name}: ${duration.toFixed(2)}ms`);

  return { result, duration };
};

// Error handling
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
});

console.log('🧪 Backend test suite initialized');
console.log(`📊 Coverage target: 95%+`);
console.log(`⚡ Test timeout: 30000ms\n`);
