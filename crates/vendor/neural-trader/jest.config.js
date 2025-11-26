module.exports = {
  testEnvironment: 'node',
  testMatch: ['**/tests/**/*.test.js'],
  testTimeout: 60000,
  verbose: true,
  collectCoverage: false,
  coverageDirectory: 'coverage',
  coveragePathIgnorePatterns: ['/node_modules/', '/neural-trader-rust/'],
  moduleFileExtensions: ['js', 'json', 'node'],
  testPathIgnorePatterns: ['/node_modules/', '/neural-trader-rust/'],
  setupFilesAfterEnv: ['<rootDir>/tests/setup.js'],
  globals: {
    'process.env.NODE_ENV': 'test'
  }
};
