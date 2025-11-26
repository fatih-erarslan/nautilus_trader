/** @type {import('jest').Config} */
module.exports = {
  displayName: 'Agentic Accounting Tests',
  testEnvironment: 'node',
  preset: 'ts-jest',

  // Test match patterns
  testMatch: [
    '<rootDir>/unit/**/*.test.ts',
    '<rootDir>/integration/**/*.test.ts',
    '<rootDir>/e2e/**/*.test.ts',
  ],

  // Module paths
  roots: ['<rootDir>'],
  modulePaths: ['<rootDir>'],
  moduleDirectories: ['node_modules', '<rootDir>'],

  // TypeScript configuration
  transform: {
    '^.+\\.tsx?$': ['ts-jest', {
      tsconfig: {
        esModuleInterop: true,
        allowSyntheticDefaultImports: true,
        strict: true,
        skipLibCheck: true,
      }
    }]
  },

  // Coverage configuration
  collectCoverage: true,
  coverageDirectory: '<rootDir>/coverage',
  coverageReporters: ['text', 'text-summary', 'html', 'lcov', 'json'],
  collectCoverageFrom: [
    '../../packages/agentic-accounting/**/src/**/*.ts',
    '!**/*.d.ts',
    '!**/node_modules/**',
    '!**/dist/**',
    '!**/coverage/**',
  ],

  // Coverage thresholds - 90% minimum
  coverageThresholds: {
    global: {
      branches: 90,
      functions: 90,
      lines: 90,
      statements: 90,
    },
  },

  // Setup files
  setupFilesAfterEnv: ['<rootDir>/utils/test-setup.ts'],

  // Timeouts
  testTimeout: 30000, // 30 seconds for integration/e2e tests

  // Global settings
  globals: {
    'ts-jest': {
      isolatedModules: true,
    },
  },

  // Clear mocks between tests
  clearMocks: true,
  resetMocks: true,
  restoreMocks: true,

  // Verbose output
  verbose: true,

  // Detect open handles
  detectOpenHandles: true,
  forceExit: true,
};
