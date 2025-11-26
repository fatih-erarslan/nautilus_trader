# Neural Trader Testing Guide

Comprehensive guide to testing and benchmarking neural-trader examples.

## Table of Contents

- [Overview](#overview)
- [Test Framework](#test-framework)
- [Benchmarking](#benchmarking)
- [Writing Tests](#writing-tests)
- [Running Tests](#running-tests)
- [CI/CD Integration](#cicd-integration)
- [Best Practices](#best-practices)

## Overview

The neural-trader project includes comprehensive testing infrastructure:

- **@neural-trader/test-framework**: Generic test utilities, mocks, and fixtures
- **@neural-trader/benchmarks**: Performance benchmarking with regression detection
- **34+ Example Tests**: Unit, integration, and benchmark tests across examples

### Test Coverage Goals

- **Unit Tests**: >85% code coverage
- **Integration Tests**: All swarm coordination and self-learning loops
- **Benchmark Tests**: Performance tracking for all critical paths
- **OpenRouter Tests**: API integration validation

## Test Framework

### Installation

```bash
cd packages/examples/test-framework
npm install
npm run build
```

### Key Features

1. **Test Utilities**
   - Environment setup/teardown
   - Random data generation
   - Metrics calculation
   - Async helpers (timeout, retry, debounce)

2. **Mocks**
   - AgentDB mock with vector storage
   - OpenRouter mock with configurable responses
   - Predictor mock with multiple models
   - Swarm mock for multi-agent testing

3. **Fixtures**
   - Market data (OHLCV, order books, ticks)
   - Trading data (signals, portfolios, trades)
   - Time series (patterns, anomalies, multivariate)

4. **Custom Matchers**
   - `toBeWithinPercent(expected, percent)`
   - `toBeValidPrediction()`
   - `toBeValidTimeSeries()`
   - `toHaveConverged(threshold)`
   - `toHaveImproved(baseline)`

### Example Usage

```typescript
import {
  setupTestEnvironment,
  createMockAgentDB,
  generateMarketData,
  installMatchers
} from '@neural-trader/test-framework';

describe('Trading Strategy', () => {
  let mockDB: any;

  beforeAll(() => {
    setupTestEnvironment({ timeout: 30000 });
    installMatchers();
  });

  beforeEach(() => {
    mockDB = createMockAgentDB();
  });

  test('predicts market movement', async () => {
    const data = generateMarketData('AAPL', 100);
    const prediction = await strategy.predict(data);

    expect(prediction).toBeValidPrediction();
    expect(prediction).toBeWithinPercent(0.75, 10);
  });
});
```

## Benchmarking

### Installation

```bash
cd packages/examples/benchmarks
npm install
npm run build
```

### Key Features

1. **Statistical Analysis**
   - Mean, median, std dev, percentiles
   - Confidence intervals
   - Outlier detection
   - Trend analysis

2. **Regression Detection**
   - Configurable thresholds
   - Statistical significance testing
   - Historical trend analysis
   - Severity classification

3. **Memory Leak Detection**
   - Linear regression on memory samples
   - Leak rate calculation
   - Time-to-OOM prediction
   - Confidence scoring

4. **Performance Profiling**
   - Hotspot identification
   - Call count tracking
   - Bottleneck detection
   - Optimization recommendations

5. **Reporting**
   - Console (colored output with tables)
   - JSON (machine-readable)
   - HTML (interactive charts)

### Example Usage

```typescript
import { BenchmarkRunner } from '@neural-trader/benchmarks';

const runner = new BenchmarkRunner({
  name: 'OrderBook Processing',
  iterations: 100,
  warmupIterations: 10
});

const result = await runner.run(async () => {
  await processOrderBook(data);
});

console.log(`Mean: ${result.mean}ms`);
console.log(`Throughput: ${result.throughput} ops/sec`);
```

### CLI Usage

```bash
# Run benchmarks
neural-bench run ./benchmarks/suite.js -i 100 -o html -f report.html

# Compare implementations
neural-bench compare baseline.js current.js

# Detect memory leaks
neural-bench memory-leak suite.js -i 100 -t 1024

# View history (requires AgentDB)
neural-bench history my-benchmark -l 20
```

## Writing Tests

### Unit Tests

Test individual components in isolation:

```typescript
describe('OrderBookAnalyzer', () => {
  let analyzer: OrderBookAnalyzer;
  let mockDB: any;

  beforeEach(() => {
    mockDB = createMockAgentDB();
    analyzer = new OrderBookAnalyzer(mockDB);
  });

  test('calculates spread correctly', async () => {
    const orderBook = generateOrderBook('AAPL', 150, 10);
    const spread = await analyzer.calculateSpread(orderBook);

    expect(spread).toBeGreaterThan(0);
    expect(spread).toBeLessThan(1);
  });

  test('handles empty order book', async () => {
    const empty = { bids: [], asks: [], timestamp: Date.now() };

    await expect(analyzer.processSnapshot(empty))
      .rejects.toThrow('Empty order book');
  });
});
```

### Integration Tests

Test component interactions and swarm coordination:

```typescript
describe('Swarm Integration', () => {
  let swarm: any;

  beforeEach(() => {
    swarm = createMockSwarm({ delay: 10 });
  });

  test('coordinates multiple agents', async () => {
    await swarm.addAgent({ id: 'agent-1', role: 'analyzer', state: {} });
    await swarm.addAgent({ id: 'agent-2', role: 'predictor', state: {} });

    const result = await swarm.coordinate(10);

    expect(result.consensusReached).toBe(true);
    expect(result.averageQuality).toBeGreaterThan(0.5);
  });

  test('shares patterns between agents', async () => {
    await swarm.addAgent({ id: 'teacher', role: 'learner', state: {} });
    await swarm.addAgent({ id: 'student', role: 'learner', state: {} });

    await swarm.sendMessage({
      from: 'teacher',
      to: 'student',
      type: 'share-patterns',
      payload: { patterns: ['pattern-1'] }
    });

    const messages = swarm.getMessages();
    expect(messages).toHaveLength(1);
  });
});
```

### Benchmark Tests

Test performance characteristics:

```typescript
describe('Performance Benchmarks', () => {
  test('processes order book quickly', async () => {
    const analyzer = new OrderBookAnalyzer(mockDB);
    const orderBook = generateOrderBook('AAPL', 150, 20);

    const runner = new BenchmarkRunner({
      name: 'OrderBook-Process',
      iterations: 100
    });

    const result = await runner.run(async () => {
      await analyzer.processSnapshot(orderBook);
    });

    expect(result.mean).toBeLessThan(50); // < 50ms
    expect(result.throughput).toBeGreaterThan(20); // > 20 ops/sec
  });

  test('scales with data size', async () => {
    const sizes = [10, 20, 50, 100];
    const results = [];

    for (const size of sizes) {
      const orderBook = generateOrderBook('AAPL', 150, size);
      const runner = new BenchmarkRunner({
        name: `Size-${size}`,
        iterations: 50
      });

      const result = await runner.run(async () => {
        await analyzer.processSnapshot(orderBook);
      });

      results.push({ size, mean: result.mean });
    }

    // Verify sub-linear scaling
    const firstMean = results[0].mean;
    const lastMean = results[results.length - 1].mean;
    const sizeRatio = sizes[sizes.length - 1] / sizes[0];
    const timeRatio = lastMean / firstMean;

    expect(timeRatio).toBeLessThan(sizeRatio);
  });
});
```

### OpenRouter Integration Tests

Test LLM integration:

```typescript
describe('OpenRouter Integration', () => {
  let mockRouter: any;

  beforeEach(() => {
    mockRouter = createMockOpenRouter({
      responses: [
        'Bullish trend detected with high confidence',
        'Market consolidation expected',
        'Bearish reversal pattern forming'
      ]
    });
  });

  test('analyzes market with LLM', async () => {
    const analyzer = new LLMMarketAnalyzer(mockRouter);
    const marketData = generateMarketData('AAPL', 100);

    const analysis = await analyzer.analyze(marketData);

    expect(analysis).toHaveProperty('sentiment');
    expect(analysis).toHaveProperty('confidence');
    expect(mockRouter.getCallCount()).toBeGreaterThan(0);
  });

  test('handles rate limits gracefully', async () => {
    mockRouter = createMockOpenRouter({ errorRate: 0.5 });

    const analyzer = new LLMMarketAnalyzer(mockRouter);

    await expect(
      analyzer.analyze(marketData, { retries: 3 })
    ).resolves.toBeDefined();
  });
});
```

## Running Tests

### Run All Tests

```bash
# From repository root
npm test

# With coverage
npm run test:coverage

# Watch mode
npm run test:watch
```

### Run Specific Example Tests

```bash
# Navigate to example
cd packages/examples/market-microstructure

# Run tests
npm test

# With coverage
npm run test:coverage

# Specific test file
npm test -- order-book-analyzer.test.ts
```

### Run Benchmarks

```bash
# Run all benchmarks
npm run benchmark

# Quick benchmarks (fewer iterations)
npm run benchmark:quick

# Export results
npm run benchmark:report
```

### Memory Leak Detection

```bash
# Run with GC exposed
node --expose-gc tests/memory-leak.test.js

# Or use npm script
npm run test:memory
```

## CI/CD Integration

### GitHub Actions

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install dependencies
        run: npm install

      - name: Run linter
        run: npm run lint

      - name: Run type check
        run: npm run typecheck

      - name: Run tests
        run: npm run test:coverage

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage/coverage-final.json

  benchmark:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install dependencies
        run: npm install

      - name: Run benchmarks
        run: npm run benchmark

      - name: Check for regressions
        run: node scripts/check-regressions.js

      - name: Upload benchmark results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: results/
```

### Regression Check Script

Create `scripts/check-regressions.js`:

```javascript
const fs = require('fs');
const { RegressionDetector } = require('@neural-trader/benchmarks');

const detector = new RegressionDetector({
  mean: 10,
  p95: 15,
  throughput: 10,
  memory: 20
});

// Load baseline and current results
const baseline = JSON.parse(fs.readFileSync('baseline.json'));
const current = JSON.parse(fs.readFileSync('results/current.json'));

const alerts = [];

for (let i = 0; i < baseline.benchmarks.length; i++) {
  const baselineResult = baseline.benchmarks[i];
  const currentResult = current.benchmarks.find(b => b.name === baselineResult.name);

  if (currentResult) {
    const regressions = detector.detect(baselineResult, currentResult);
    alerts.push(...regressions);
  }
}

if (alerts.length > 0) {
  console.error(`❌ ${alerts.length} performance regressions detected!`);

  for (const alert of alerts) {
    console.error(`[${alert.severity.toUpperCase()}] ${alert.benchmark}`);
    console.error(`  ${alert.metric}: ${alert.actual.toFixed(2)}% (threshold: ${alert.threshold}%)`);
  }

  process.exit(1);
} else {
  console.log('✅ No performance regressions detected');
}
```

### Pre-commit Hooks

Create `.husky/pre-commit`:

```bash
#!/bin/sh
. "$(dirname "$0")/_/husky.sh"

# Run tests
npm run test

# Run linter
npm run lint

# Type check
npm run typecheck
```

## Best Practices

### 1. Test Organization

```
example/
├── src/
│   ├── analyzer.ts
│   └── predictor.ts
├── tests/
│   ├── analyzer.test.ts          # Unit tests
│   ├── predictor.test.ts         # Unit tests
│   ├── integration.test.ts       # Integration tests
│   ├── swarm-integration.test.ts # Swarm tests
│   └── benchmark.test.ts         # Performance tests
└── package.json
```

### 2. Test Isolation

- Each test should be independent
- Use `beforeEach()` to reset state
- Clean up resources in `afterEach()`
- Don't share mutable state between tests

### 3. Performance Testing

- Always use warmup iterations
- Run benchmarks in consistent environments
- Track historical performance
- Set realistic thresholds

### 4. Code Coverage

- Aim for >85% coverage
- Focus on critical paths
- Test edge cases and error handling
- Don't chase 100% coverage

### 5. Mock Usage

- Mock external dependencies (APIs, databases)
- Use realistic mock data
- Test with both success and failure scenarios
- Verify mock call counts

### 6. Async Testing

- Always return promises or use async/await
- Set appropriate timeouts
- Test concurrent operations
- Handle race conditions

### 7. Error Testing

- Test error paths explicitly
- Verify error messages
- Check error recovery
- Test graceful degradation

### 8. Documentation

- Document test setup requirements
- Explain complex test scenarios
- Include example usage
- Keep README up to date

## Common Patterns

### Testing Self-Learning

```typescript
test('improves with training', async () => {
  const learner = new PatternLearner(mockDB, mockPredictor);

  // Initial performance
  await learner.train(trainingData.slice(0, 100));
  const accuracy1 = learner.getModelAccuracy();

  // After more training
  await learner.train(trainingData.slice(100, 200));
  const accuracy2 = learner.getModelAccuracy();

  expect(accuracy2).toBeGreaterThanOrEqual(accuracy1 * 0.95);
});
```

### Testing Swarm Coordination

```typescript
test('reaches consensus', async () => {
  // Setup agents
  for (let i = 0; i < 5; i++) {
    await swarm.addAgent({
      id: `agent-${i}`,
      role: 'predictor',
      state: {}
    });
  }

  // Run coordination
  const result = await swarm.coordinate(10);

  expect(result.consensusReached).toBe(true);
  expect(result.convergenceTime).toBeGreaterThan(0);
});
```

### Testing Memory Efficiency

```typescript
test('manages memory efficiently', async () => {
  const detector = new MemoryLeakDetector();

  const result = await detector.test(async () => {
    await processLargeDataset();
  }, 100);

  expect(result.leaked).toBe(false);
  expect(result.analysis.trend).toBe('stable');
});
```

## Troubleshooting

### Tests Timing Out

- Increase timeout: `setupTestEnvironment({ timeout: 60000 })`
- Check for unhandled promises
- Verify async operations complete

### Flaky Tests

- Ensure test isolation
- Check for race conditions
- Use `waitForCondition()` instead of fixed delays
- Verify mock behavior

### Low Coverage

- Check ignored files in `jest.config.js`
- Run with `--coverage` flag
- Use coverage report to find gaps

### Memory Leaks in Tests

- Clean up resources in `afterEach()`
- Clear mocks and caches
- Run with `--detectLeaks` flag

## Resources

- [Jest Documentation](https://jestjs.io/)
- [Testing Best Practices](https://testingjavascript.com/)
- [AgentDB Documentation](https://github.com/oomol/agentdb)
- [Neural Trader Repository](https://github.com/ruvnet/neural-trader)

## Support

For issues and questions:

- GitHub Issues: https://github.com/ruvnet/neural-trader/issues
- Documentation: https://github.com/ruvnet/neural-trader/tree/main/docs

## License

MIT OR Apache-2.0
