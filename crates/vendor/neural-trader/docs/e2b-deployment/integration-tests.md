# E2B Strategy Integration Tests

## Overview

This document provides integration test patterns for E2B-deployed trading strategies.

## Test Setup

### Prerequisites

```bash
npm install --save-dev jest supertest @e2b/sdk
```

### Test Configuration

Create `jest.config.js`:

```javascript
module.exports = {
  testEnvironment: 'node',
  testTimeout: 60000, // 60 seconds for E2B operations
  setupFilesAfterEnv: ['./tests/setup.js']
};
```

## Test Patterns

### 1. Sandbox Creation and Health Check

```javascript
const { E2B } = require('@e2b/sdk');
const request = require('supertest');

describe('Momentum Strategy E2B Deployment', () => {
  let sandbox;
  let baseUrl;

  beforeAll(async () => {
    // Create sandbox
    sandbox = await E2B.Sandbox.create({
      template: 'node',
      env_vars: {
        ALPACA_API_KEY: process.env.ALPACA_API_KEY,
        ALPACA_SECRET_KEY: process.env.ALPACA_SECRET_KEY,
        PORT: '3000'
      }
    });

    baseUrl = `http://${sandbox.getHost()}:3000`;

    // Upload strategy files
    await sandbox.filesystem.write('/app/index.js',
      fs.readFileSync('./e2b-strategies/momentum/index.js', 'utf-8')
    );

    await sandbox.filesystem.write('/app/package.json',
      fs.readFileSync('./e2b-strategies/momentum/package.json', 'utf-8')
    );

    // Install and start
    const install = await sandbox.process.start('cd /app && npm install');
    await install.wait();

    const start = await sandbox.process.startAndWait('cd /app && npm start &');

    // Wait for server to start
    await new Promise(resolve => setTimeout(resolve, 5000));
  });

  afterAll(async () => {
    if (sandbox) {
      await sandbox.close();
    }
  });

  test('Health check returns healthy status', async () => {
    const response = await request(baseUrl)
      .get('/health')
      .expect(200);

    expect(response.body).toMatchObject({
      status: 'healthy',
      strategy: 'momentum'
    });
  });

  test('Status endpoint returns account info', async () => {
    const response = await request(baseUrl)
      .get('/status')
      .expect(200);

    expect(response.body).toHaveProperty('account');
    expect(response.body.account).toHaveProperty('equity');
    expect(response.body.account).toHaveProperty('cash');
  });

  test('Manual execution triggers strategy', async () => {
    const response = await request(baseUrl)
      .post('/execute')
      .expect(200);

    expect(response.body).toMatchObject({
      success: true,
      message: 'Strategy executed'
    });
  });
});
```

### 2. Neural Forecast Strategy Tests

```javascript
describe('Neural Forecast Strategy E2B Deployment', () => {
  let sandbox;
  let baseUrl;

  beforeAll(async () => {
    sandbox = await E2B.Sandbox.create({
      template: 'node',
      env_vars: {
        ALPACA_API_KEY: process.env.ALPACA_API_KEY,
        ALPACA_SECRET_KEY: process.env.ALPACA_SECRET_KEY,
        PORT: '3001'
      },
      memory_mb: 2048  // More memory for TensorFlow
    });

    baseUrl = `http://${sandbox.getHost()}:3001`;

    // Upload and start
    await uploadAndStart(sandbox, 'neural-forecast');
    await new Promise(resolve => setTimeout(resolve, 10000)); // Wait for TF init
  });

  afterAll(async () => {
    if (sandbox) await sandbox.close();
  });

  test('Model training endpoint works', async () => {
    const response = await request(baseUrl)
      .post('/retrain/AAPL')
      .expect(200);

    expect(response.body).toMatchObject({
      success: true,
      symbol: 'AAPL',
      message: 'Model retrained'
    });
  });

  test('Status shows loaded models', async () => {
    const response = await request(baseUrl)
      .get('/status')
      .expect(200);

    expect(response.body).toHaveProperty('models');
    expect(Array.isArray(response.body.models)).toBe(true);
  });

  test('Health check shows neural-forecast strategy', async () => {
    const response = await request(baseUrl)
      .get('/health')
      .expect(200);

    expect(response.body.strategy).toBe('neural-forecast');
    expect(response.body.modelsLoaded).toBeDefined();
  });
});
```

### 3. Risk Manager Tests

```javascript
describe('Risk Manager E2B Deployment', () => {
  let sandbox;
  let baseUrl;

  beforeAll(async () => {
    sandbox = await E2B.Sandbox.create({
      template: 'node',
      env_vars: {
        ALPACA_API_KEY: process.env.ALPACA_API_KEY,
        ALPACA_SECRET_KEY: process.env.ALPACA_SECRET_KEY,
        PORT: '3003'
      }
    });

    baseUrl = `http://${sandbox.getHost()}:3003`;
    await uploadAndStart(sandbox, 'risk-manager');
    await new Promise(resolve => setTimeout(resolve, 5000));
  });

  afterAll(async () => {
    if (sandbox) await sandbox.close();
  });

  test('Risk metrics endpoint returns VaR and CVaR', async () => {
    // Trigger monitoring first
    await request(baseUrl).post('/monitor');

    const response = await request(baseUrl)
      .get('/metrics')
      .expect(200);

    expect(response.body).toHaveProperty('var');
    expect(response.body).toHaveProperty('cvar');
    expect(response.body).toHaveProperty('maxDrawdown');
    expect(response.body).toHaveProperty('sharpeRatio');
  });

  test('Alerts endpoint returns alert array', async () => {
    const response = await request(baseUrl)
      .get('/alerts')
      .expect(200);

    expect(response.body).toHaveProperty('alerts');
    expect(Array.isArray(response.body.alerts)).toBe(true);
  });

  test('Manual monitoring execution works', async () => {
    const response = await request(baseUrl)
      .post('/monitor')
      .expect(200);

    expect(response.body.success).toBe(true);
    expect(response.body.metrics).toBeDefined();
  });
});
```

### 4. Portfolio Optimizer Tests

```javascript
describe('Portfolio Optimizer E2B Deployment', () => {
  let sandbox;
  let baseUrl;

  beforeAll(async () => {
    sandbox = await E2B.Sandbox.create({
      template: 'node',
      env_vars: {
        ALPACA_API_KEY: process.env.ALPACA_API_KEY,
        ALPACA_SECRET_KEY: process.env.ALPACA_SECRET_KEY,
        PORT: '3004'
      }
    });

    baseUrl = `http://${sandbox.getHost()}:3004`;
    await uploadAndStart(sandbox, 'portfolio-optimizer');
    await new Promise(resolve => setTimeout(resolve, 5000));
  });

  afterAll(async () => {
    if (sandbox) await sandbox.close();
  });

  test('Optimization endpoint calculates allocations', async () => {
    const response = await request(baseUrl)
      .post('/optimize')
      .expect(200);

    expect(response.body.success).toBe(true);
    expect(response.body.result).toHaveProperty('allocations');
    expect(response.body.result).toHaveProperty('stats');
  });

  test('Status shows target and current allocations', async () => {
    // Run optimization first
    await request(baseUrl).post('/optimize');

    const response = await request(baseUrl)
      .get('/status')
      .expect(200);

    expect(response.body.targetAllocations).toBeDefined();
    expect(response.body.expectedReturn).toBeDefined();
    expect(response.body.sharpeRatio).toBeDefined();
  });

  test('Rebalance endpoint executes trades', async () => {
    // Optimize first to set targets
    await request(baseUrl).post('/optimize');

    const response = await request(baseUrl)
      .post('/rebalance')
      .expect(200);

    expect(response.body.success).toBe(true);
  });
});
```

### 5. Mean Reversion Tests

```javascript
describe('Mean Reversion Strategy E2B Deployment', () => {
  let sandbox;
  let baseUrl;

  beforeAll(async () => {
    sandbox = await E2B.Sandbox.create({
      template: 'node',
      env_vars: {
        ALPACA_API_KEY: process.env.ALPACA_API_KEY,
        ALPACA_SECRET_KEY: process.env.ALPACA_SECRET_KEY,
        PORT: '3002'
      }
    });

    baseUrl = `http://${sandbox.getHost()}:3002`;
    await uploadAndStart(sandbox, 'mean-reversion');
    await new Promise(resolve => setTimeout(resolve, 5000));
  });

  afterAll(async () => {
    if (sandbox) await sandbox.close();
  });

  test('Statistics endpoint returns z-scores', async () => {
    // Execute strategy first to calculate stats
    await request(baseUrl).post('/execute');

    const response = await request(baseUrl)
      .get('/statistics/GLD')
      .expect(200);

    expect(response.body).toHaveProperty('symbol', 'GLD');
    expect(response.body).toHaveProperty('zScore');
    expect(response.body).toHaveProperty('sma');
    expect(response.body).toHaveProperty('stdDev');
  });

  test('Status includes statistics for all symbols', async () => {
    await request(baseUrl).post('/execute');

    const response = await request(baseUrl)
      .get('/status')
      .expect(200);

    expect(response.body.statistics).toBeDefined();
    expect(Array.isArray(response.body.statistics)).toBe(true);
  });
});
```

## Multi-Strategy Integration Tests

### Full System Test

```javascript
describe('Multi-Strategy E2B Deployment', () => {
  const strategies = [
    { name: 'momentum', port: 3000 },
    { name: 'neural-forecast', port: 3001 },
    { name: 'mean-reversion', port: 3002 },
    { name: 'risk-manager', port: 3003 },
    { name: 'portfolio-optimizer', port: 3004 }
  ];

  let sandboxes = {};
  let baseUrls = {};

  beforeAll(async () => {
    // Deploy all strategies in parallel
    await Promise.all(strategies.map(async (strategy) => {
      const sandbox = await E2B.Sandbox.create({
        template: 'node',
        env_vars: {
          ALPACA_API_KEY: process.env.ALPACA_API_KEY,
          ALPACA_SECRET_KEY: process.env.ALPACA_SECRET_KEY,
          PORT: strategy.port.toString()
        },
        memory_mb: strategy.name === 'neural-forecast' ? 2048 : 512
      });

      await uploadAndStart(sandbox, strategy.name);

      sandboxes[strategy.name] = sandbox;
      baseUrls[strategy.name] = `http://${sandbox.getHost()}:${strategy.port}`;
    }));

    // Wait for all to initialize
    await new Promise(resolve => setTimeout(resolve, 15000));
  });

  afterAll(async () => {
    await Promise.all(
      Object.values(sandboxes).map(sandbox => sandbox.close())
    );
  });

  test('All strategies are healthy', async () => {
    const healthChecks = await Promise.all(
      strategies.map(async (strategy) => {
        const response = await request(baseUrls[strategy.name])
          .get('/health');
        return { name: strategy.name, status: response.status };
      })
    );

    healthChecks.forEach(check => {
      expect(check.status).toBe(200);
    });
  });

  test('Trading workflow: optimize -> trade -> monitor', async () => {
    // 1. Optimize portfolio
    const optimizeResponse = await request(baseUrls['portfolio-optimizer'])
      .post('/optimize');
    expect(optimizeResponse.body.success).toBe(true);

    // 2. Execute trading strategies
    await Promise.all([
      request(baseUrls['momentum']).post('/execute'),
      request(baseUrls['neural-forecast']).post('/execute'),
      request(baseUrls['mean-reversion']).post('/execute')
    ]);

    // 3. Monitor risk
    const riskResponse = await request(baseUrls['risk-manager'])
      .post('/monitor');
    expect(riskResponse.body.success).toBe(true);
  });

  test('Risk manager monitors all positions', async () => {
    // Execute some trades first
    await request(baseUrls['momentum']).post('/execute');

    // Check risk metrics
    const response = await request(baseUrls['risk-manager'])
      .get('/metrics');

    expect(response.body.positions).toBeDefined();
    expect(response.body.var).toBeDefined();
  });
});
```

## Helper Functions

```javascript
// helpers.js
const fs = require('fs');

async function uploadAndStart(sandbox, strategyName) {
  const basePath = `./e2b-strategies/${strategyName}`;

  // Upload files
  await sandbox.filesystem.write('/app/index.js',
    fs.readFileSync(`${basePath}/index.js`, 'utf-8')
  );

  await sandbox.filesystem.write('/app/package.json',
    fs.readFileSync(`${basePath}/package.json`, 'utf-8')
  );

  // Install dependencies
  const install = await sandbox.process.start('cd /app && npm install');
  await install.wait();

  // Start in background
  await sandbox.process.start('cd /app && npm start &');
}

async function checkLogs(sandbox, searchTerm) {
  const logs = await sandbox.process.startAndWait(
    `grep "${searchTerm}" /app/logs/*.log`
  );
  return logs.stdout;
}

async function waitForHealthy(baseUrl, maxRetries = 10) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const response = await request(baseUrl).get('/health');
      if (response.status === 200) return true;
    } catch (error) {
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  }
  throw new Error('Strategy failed to become healthy');
}

module.exports = { uploadAndStart, checkLogs, waitForHealthy };
```

## Running Tests

```bash
# Run all integration tests
npm test -- --testPathPattern=integration

# Run specific strategy tests
npm test -- momentum.integration.test.js

# Run with coverage
npm test -- --coverage

# Run in watch mode
npm test -- --watch
```

## Continuous Integration

```yaml
# .github/workflows/e2b-integration.yml
name: E2B Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install dependencies
        run: npm ci

      - name: Run integration tests
        env:
          ALPACA_API_KEY: ${{ secrets.ALPACA_API_KEY }}
          ALPACA_SECRET_KEY: ${{ secrets.ALPACA_SECRET_KEY }}
        run: npm test -- --testPathPattern=integration
```

## Performance Benchmarks

```javascript
describe('Performance Benchmarks', () => {
  test('Strategy execution completes within 30 seconds', async () => {
    const start = Date.now();
    await request(baseUrl).post('/execute');
    const duration = Date.now() - start;

    expect(duration).toBeLessThan(30000);
  });

  test('Health check responds within 1 second', async () => {
    const start = Date.now();
    await request(baseUrl).get('/health');
    const duration = Date.now() - start;

    expect(duration).toBeLessThan(1000);
  });
});
```

## Troubleshooting Tests

### Common Issues

1. **Timeout errors**: Increase `jest.config.js` timeout
2. **Port conflicts**: Ensure unique ports per strategy
3. **Memory errors (TensorFlow)**: Increase sandbox memory
4. **API rate limits**: Add delays between tests
5. **Market closed**: Mock Alpaca responses or test during market hours
