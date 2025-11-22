module.exports = {
  displayName: 'CWTS Comprehensive Test Suite',
  testEnvironment: 'node',
  roots: ['<rootDir>/tests'],
  testMatch: [
    '**/tests/**/*.test.js',
    '**/tests/**/*.test.ts',
    '**/tests/**/*.spec.js',
    '**/tests/**/*.spec.ts'
  ],
  coverageDirectory: 'coverage',
  collectCoverageFrom: [
    'src/**/*.{js,ts}',
    'quantum_trading/**/*.{js,ts}',
    '!src/**/*.d.ts',
    '!src/**/*.config.js',
    '!**/__tests__/**',
    '!**/node_modules/**'
  ],
  coverageReporters: [
    'text',
    'text-summary',
    'html',
    'lcov',
    'json-summary'
  ],
  coverageThreshold: {
    global: {
      branches: 100,
      functions: 100,
      lines: 100,
      statements: 100
    }
  },
  setupFilesAfterEnv: ['<rootDir>/tests/utils/test-setup.js'],
  testTimeout: 30000,
  maxWorkers: '50%',
  verbose: true,
  bail: false,
  errorOnDeprecated: true,
  projects: [
    {
      displayName: 'Unit Tests',
      testMatch: ['<rootDir>/tests/unit_tests/**/*.test.{js,ts}'],
      coverageDirectory: 'coverage/unit'
    },
    {
      displayName: 'Integration Tests',
      testMatch: ['<rootDir>/tests/integration_tests/**/*.test.{js,ts}'],
      coverageDirectory: 'coverage/integration',
      testTimeout: 60000
    },
    {
      displayName: 'Property Tests',
      testMatch: ['<rootDir>/tests/property_tests/**/*.test.{js,ts}'],
      coverageDirectory: 'coverage/property',
      testTimeout: 120000
    },
    {
      displayName: 'Stress Tests',
      testMatch: ['<rootDir>/tests/stress_tests/**/*.test.{js,ts}'],
      coverageDirectory: 'coverage/stress',
      testTimeout: 300000
    },
    {
      displayName: 'Chaos Tests',
      testMatch: ['<rootDir>/tests/chaos_tests/**/*.test.{js,ts}'],
      coverageDirectory: 'coverage/chaos',
      testTimeout: 180000
    }
  ]
};