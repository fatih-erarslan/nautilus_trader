/**
 * Jest configuration for CLI tests
 */

module.exports = {
  displayName: 'cli-tests',
  testEnvironment: 'node',
  testMatch: [
    '**/tests/cli/unit/**/*.test.js',
    '**/tests/cli/integration/**/*.test.js',
    '**/tests/cli/e2e/**/*.test.js',
    '**/tests/cli/performance/**/*.test.js'
  ],
  testPathIgnorePatterns: [
    '/node_modules/',
    '/neural-trader-rust/'
  ],
  coverageDirectory: 'coverage/cli',
  coveragePathIgnorePatterns: [
    '/node_modules/',
    '/neural-trader-rust/',
    '/tests/cli/__mocks__/',
    '/tests/cli/fixtures/'
  ],
  collectCoverageFrom: [
    'bin/cli.js',
    '!bin/cli-v2.3.11.js',
    '!bin/cli.js.old'
  ],
  coverageThreshold: {
    global: {
      statements: 80,
      branches: 75,
      functions: 80,
      lines: 80
    }
  },
  testTimeout: 30000,
  verbose: true,
  moduleFileExtensions: ['js', 'json', 'node'],
  globals: {
    'process.env.NODE_ENV': 'test'
  },
  // Test sequencer for better organization
  testSequencer: './test-sequencer.js',
  // Setup file
  setupFilesAfterEnv: ['<rootDir>/setup-cli-tests.js'],
  // Mock modules
  moduleNameMapper: {
    '^fs$': '<rootDir>/__mocks__/fs.js',
    '^child_process$': '<rootDir>/__mocks__/child_process.js'
  }
};
