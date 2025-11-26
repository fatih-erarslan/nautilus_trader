/**
 * Jest Configuration for Neural Trader Backend Tests
 * Comprehensive test configuration with coverage reporting
 */

module.exports = {
  // Test environment
  testEnvironment: 'node',

  // Test match patterns
  testMatch: [
    '**/tests/backend/**/*.test.js'
  ],

  // Coverage configuration
  collectCoverage: true,
  coverageDirectory: 'coverage/backend',
  coverageReporters: ['text', 'lcov', 'html', 'json-summary'],

  // Coverage thresholds (95%+ target)
  coverageThresholds: {
    global: {
      branches: 95,
      functions: 95,
      lines: 95,
      statements: 95
    }
  },

  // Files to collect coverage from
  collectCoverageFrom: [
    'neural-trader-rust/packages/neural-trader-backend/**/*.js',
    '!**node_modules/**',
    '!**/tests/**',
    '!**/coverage/**'
  ],

  // Test timeout (increased for performance tests)
  testTimeout: 30000,

  // Setup files
  setupFilesAfterEnv: ['<rootDir>/tests/backend/setup.js'],

  // Verbose output
  verbose: true,

  // Detect open handles (useful for debugging)
  detectOpenHandles: false,

  // Force exit after tests complete
  forceExit: true,

  // Maximum workers (parallel test execution)
  maxWorkers: '50%',

  // Clear mocks between tests
  clearMocks: true,
  resetMocks: true,
  restoreMocks: true,

  // Module paths
  moduleDirectories: ['node_modules', '<rootDir>'],

  // Global setup/teardown
  // globalSetup: '<rootDir>/tests/backend/global-setup.js',
  // globalTeardown: '<rootDir>/tests/backend/global-teardown.js',

  // Transform files (if using TypeScript/Babel)
  // transform: {
  //   '^.+\\.js$': 'babel-jest',
  // },

  // Test results processor
  // testResultsProcessor: 'jest-sonar-reporter',

  // Reporter configuration
  reporters: [
    'default',
    [
      'jest-html-reporter',
      {
        pageTitle: 'Neural Trader Backend Test Report',
        outputPath: 'coverage/backend/test-report.html',
        includeFailureMsg: true,
        includeConsoleLog: true,
        sort: 'status',
        theme: 'darkTheme'
      }
    ]
  ],

  // Error handling
  bail: false, // Continue running tests after failure

  // Watch plugins (for development)
  watchPlugins: [
    'jest-watch-typeahead/filename',
    'jest-watch-typeahead/testname'
  ],

  // Coverage path ignore patterns
  coveragePathIgnorePatterns: [
    '/node_modules/',
    '/tests/',
    '/coverage/',
    '/dist/',
    '/build/'
  ]
};
