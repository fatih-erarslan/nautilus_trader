/**
 * Mock data and fixtures for CLI testing
 */

const mockPackageJson = {
  name: 'test-project',
  version: '1.0.0',
  dependencies: {
    'neural-trader': '^2.3.15',
    '@neural-trader/core': '^1.0.1'
  }
};

const mockConfigJson = {
  trading: {
    provider: 'alpaca',
    symbols: ['AAPL', 'MSFT'],
    strategy: 'momentum',
    parameters: {
      threshold: 0.02,
      lookback: 20,
      stop_loss: 0.05
    }
  },
  risk: {
    max_position_size: 10000,
    max_portfolio_risk: 0.02,
    stop_loss_pct: 0.05
  }
};

const mockNAPIBindings = {
  fetchMarketData: jest.fn().mockResolvedValue([
    { timestamp: '2024-01-01', open: 100, close: 105 },
    { timestamp: '2024-01-02', open: 105, close: 110 }
  ]),
  runStrategy: jest.fn().mockResolvedValue({
    signals: ['BUY', 'HOLD'],
    profit: 1500.50
  }),
  backtest: jest.fn().mockResolvedValue({
    total_return: 0.15,
    sharpe_ratio: 1.8,
    max_drawdown: 0.12
  }),
  trainModel: jest.fn().mockResolvedValue({
    accuracy: 0.92,
    loss: 0.08
  }),
  predict: jest.fn().mockResolvedValue({
    prediction: 'UP',
    confidence: 0.87
  })
};

const mockProcessEnv = {
  NODE_ENV: 'test',
  PATH: '/usr/bin:/bin',
  HOME: '/home/test'
};

const mockFileSystem = {
  'package.json': JSON.stringify(mockPackageJson, null, 2),
  'config.json': JSON.stringify(mockConfigJson, null, 2),
  'README.md': '# Test Project\n\nTest readme content',
  'src/main.js': 'console.log("Hello from Neural Trader!");'
};

module.exports = {
  mockPackageJson,
  mockConfigJson,
  mockNAPIBindings,
  mockProcessEnv,
  mockFileSystem
};
