module.exports = {
  testEnvironment: 'node',
  testMatch: [
    '**/tests/midstreamer/**/*.test.js'
  ],
  coverageDirectory: 'coverage/midstreamer',
  collectCoverageFrom: [
    'src/midstreamer/**/*.js',
    '!src/midstreamer/**/*.test.js'
  ],
  coverageThresholds: {
    global: {
      statements: 80,
      branches: 75,
      functions: 80,
      lines: 80
    }
  },
  testTimeout: 10000,
  verbose: true,
  bail: false,
  maxWorkers: '50%'
};
