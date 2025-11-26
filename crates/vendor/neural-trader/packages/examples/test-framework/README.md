# @neural-trader/test-framework

Generic test utilities, mocks, and helpers for neural-trader examples.

## Features

- 🧪 **Test Utilities**: Comprehensive helper functions for common testing scenarios
- 🎭 **Mocks**: Ready-to-use mocks for AgentDB, OpenRouter, Predictor, and Swarm
- 📊 **Fixtures**: Pre-generated market data, trading signals, and time series
- ✨ **Custom Matchers**: Jest matchers for neural-trader specific assertions
- ⚡ **Async Helpers**: Utilities for testing async operations with timeouts and retries
- ⏱️ **Time Utilities**: Mock timers and time series generation

## Installation

```bash
npm install --save-dev @neural-trader/test-framework
```

## Quick Start

```typescript
import {
  setupTestEnvironment,
  cleanupTestEnvironment,
  createMockAgentDB,
  createMockOpenRouter,
  generateMarketData,
  installMatchers
} from '@neural-trader/test-framework';

describe('My Trading Strategy', () => {
  let mockDB: any;

  beforeAll(() => {
    setupTestEnvironment({
      timeout: 30000,
      mockAgentDB: true,
      mockOpenRouter: true
    });
    installMatchers(); // Install custom Jest matchers
  });

  beforeEach(() => {
    mockDB = createMockAgentDB();
  });

  afterEach(() => {
    mockDB.clear();
  });

  afterAll(() => {
    cleanupTestEnvironment();
  });

  test('should predict market movement', async () => {
    const marketData = generateMarketData('AAPL', 100);
    const prediction = await myStrategy.predict(marketData);

    expect(prediction).toBeValidPrediction();
    expect(prediction).toBeWithinPercent(0.7, 10); // Within 10% of 0.7
  });
});
```

## Test Utilities

### Setup and Teardown

```typescript
import { setupTestEnvironment, cleanupTestEnvironment } from '@neural-trader/test-framework';

setupTestEnvironment({
  timeout: 30000,
  verbose: false,
  mockOpenRouter: true,
  mockAgentDB: true
});

// Run your tests...

cleanupTestEnvironment();
```

### Test Helpers

```typescript
import {
  generateRandomData,
  calculateMetrics,
  assertWithinPercent,
  waitForCondition,
  retryWithBackoff,
  measureTime
} from '@neural-trader/test-framework';

// Generate random data
const data = generateRandomData(100, 0, 100);

// Calculate accuracy metrics
const metrics = calculateMetrics(predictions, actuals);
console.log(`Accuracy: ${metrics.accuracy}`);

// Assert within tolerance
const withinTolerance = assertWithinPercent(result, expected, 5); // 5%

// Wait for async condition
await waitForCondition(() => swarm.isConverged(), 5000);

// Retry with exponential backoff
const result = await retryWithBackoff(async () => {
  return await unreliableOperation();
}, 3);

// Measure execution time
const { result, duration } = await measureTime(async () => {
  return await expensiveOperation();
});
```

## Mocks

### AgentDB Mock

```typescript
import { createMockAgentDB } from '@neural-trader/test-framework';

const mockDB = createMockAgentDB({
  delay: 10,        // Simulate 10ms latency
  errorRate: 0.01   // 1% error rate
});

await mockDB.add('id1', [1, 2, 3], { label: 'test' });
const results = await mockDB.query({ vector: [1, 2, 3], k: 5 });

console.log(`DB size: ${mockDB.size()}`);
console.log(`Calls: ${mockDB.getCallCount()}`);
```

### OpenRouter Mock

```typescript
import { createMockOpenRouter } from '@neural-trader/test-framework';

const mockRouter = createMockOpenRouter({
  delay: 50,
  responses: [
    'Bullish market conditions',
    'Bearish trend detected',
    'Consolidation phase'
  ]
});

const response = await mockRouter.chat({
  model: 'anthropic/claude-3',
  messages: [{ role: 'user', content: 'Analyze market' }]
});

console.log(response.choices[0].message.content);
console.log(`Total calls: ${mockRouter.getCallCount()}`);
```

### Predictor Mock

```typescript
import { createMockPredictor } from '@neural-trader/test-framework';

const predictor = createMockPredictor({
  model: 'linear' // or 'random', 'constant'
});

const result = await predictor.predict({
  features: [[1, 2, 3], [4, 5, 6]],
  horizon: 5
});

console.log(result.predictions);
console.log(result.confidenceIntervals);
```

### Swarm Mock

```typescript
import { createMockSwarm } from '@neural-trader/test-framework';

const swarm = createMockSwarm({ delay: 10 });

await swarm.addAgent({ id: 'agent-1', role: 'analyzer', state: {} });
await swarm.addAgent({ id: 'agent-2', role: 'predictor', state: {} });

await swarm.broadcast('coordinator', 'update', { data: marketData });

const metrics = await swarm.coordinate(10);
console.log(`Consensus: ${metrics.consensusReached}`);
```

## Fixtures

### Market Data

```typescript
import {
  generateMarketData,
  generateOrderBook,
  generateTicks,
  SAMPLE_MARKET_DATA,
  SAMPLE_ORDER_BOOK
} from '@neural-trader/test-framework';

// Generate OHLCV data
const data = generateMarketData('AAPL', 100, {
  startPrice: 150,
  volatility: 0.02,
  trend: 0.001
});

// Generate order book
const orderBook = generateOrderBook('AAPL', 150, 20);

// Use pre-made samples
const sample = SAMPLE_MARKET_DATA;
```

### Trading Data

```typescript
import {
  generateSignals,
  generatePortfolio,
  generateTradeHistory
} from '@neural-trader/test-framework';

const signals = generateSignals('AAPL', 50, {
  buyRatio: 0.6,
  avgConfidence: 0.75
});

const portfolio = generatePortfolio(['AAPL', 'GOOGL'], 100000);

const trades = generateTradeHistory(100, ['AAPL', 'GOOGL']);
```

### Time Series

```typescript
import {
  generateTimeSeriesWithPattern,
  generateMultivariateTimeSeries,
  generateTimeSeriesWithAnomalies,
  LINEAR_SERIES,
  SINE_SERIES
} from '@neural-trader/test-framework';

// Generate specific patterns
const linear = generateTimeSeriesWithPattern(100, 'linear', {
  trend: 0.5
});

const seasonal = generateTimeSeriesWithPattern(168, 'seasonal', {
  amplitude: 10,
  trend: 0.1
});

// Multivariate series
const multivariate = generateMultivariateTimeSeries(
  100,
  ['price', 'volume', 'volatility']
);

// With anomalies
const withAnomalies = generateTimeSeriesWithAnomalies(100, 0.05, 5);
```

## Custom Matchers

```typescript
import { installMatchers } from '@neural-trader/test-framework';

beforeAll(() => {
  installMatchers();
});

test('custom matchers', () => {
  // Within percentage
  expect(100).toBeWithinPercent(105, 10); // Within 10%

  // Valid prediction
  expect(0.75).toBeValidPrediction(); // Between 0 and 1

  // Valid time series
  expect([1, 2, 3, 4, 5]).toBeValidTimeSeries();

  // Convergence
  const values = [1, 0.9, 0.85, 0.84, 0.83, 0.83];
  expect(values).toHaveConverged(0.01);

  // Improvement
  expect(0.95).toHaveImproved(0.85); // Improved over baseline
});
```

## Async Helpers

### Timeouts

```typescript
import { withTimeout } from '@neural-trader/test-framework';

const result = await withTimeout(
  slowOperation(),
  5000,
  'Operation timed out'
);
```

### Parallel Execution

```typescript
import { parallelLimit } from '@neural-trader/test-framework';

const items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

const results = await parallelLimit(items, 3, async (item) => {
  return await processItem(item);
});
```

### Debounce and Throttle

```typescript
import { debounceAsync, throttleAsync } from '@neural-trader/test-framework';

const debouncedFn = debounceAsync(expensiveOperation, 1000);
const throttledFn = throttleAsync(frequentOperation, 100);
```

## Time Utilities

### Mock Timer

```typescript
import { MockTimer } from '@neural-trader/test-framework';

const timer = new MockTimer();

timer.setTimeout(() => console.log('Fired!'), 1000);

// Advance time
timer.tick(500);  // Nothing happens
timer.tick(500);  // Callback fires

console.log(timer.now()); // Current mock time
```

### Time Series Generation

```typescript
import { generateTimestamps, generateTimeSeries, StopWatch } from '@neural-trader/test-framework';

const timestamps = generateTimestamps(100, 60000); // 100 points, 1 minute apart

const series = generateTimeSeries(100, 60000, {
  trend: 0.1,
  seasonality: 5,
  noise: 0.5
});

const stopwatch = new StopWatch();
stopwatch.start();
// ... do work ...
stopwatch.stop();
console.log(`Elapsed: ${stopwatch.elapsed()}ms`);
```

## Best Practices

### 1. Use Test Environment Setup

Always use `setupTestEnvironment()` to ensure consistent test configuration:

```typescript
beforeAll(() => {
  setupTestEnvironment({
    timeout: 30000,
    mockAgentDB: true,
    mockOpenRouter: true
  });
});
```

### 2. Clean Up Resources

Clean up mocks and test data after each test:

```typescript
afterEach(() => {
  mockDB.clear();
  mockRouter.reset();
});
```

### 3. Use Fixtures

Use pre-generated fixtures for consistent test data:

```typescript
import { SAMPLE_MARKET_DATA, SAMPLE_ORDER_BOOK } from '@neural-trader/test-framework';

test('analyze market data', () => {
  const analysis = analyze(SAMPLE_MARKET_DATA);
  expect(analysis).toBeDefined();
});
```

### 4. Test Performance

Use `measureTime()` to ensure operations meet performance requirements:

```typescript
test('processes data quickly', async () => {
  const { duration } = await measureTime(() => processData(largeDataset));
  expect(duration).toBeLessThan(1000);
});
```

### 5. Handle Async Operations

Use async helpers for reliable async testing:

```typescript
test('waits for condition', async () => {
  await waitForCondition(() => service.isReady(), 5000);
  expect(service.isReady()).toBe(true);
});
```

## TypeScript Support

Full TypeScript support with type definitions:

```typescript
import type {
  TestConfig,
  MockOptions,
  TimeSeriesData,
  MarketData,
  TradingSignal,
  TestMetrics,
  SwarmMetrics
} from '@neural-trader/test-framework';
```

## Contributing

See the main neural-trader repository for contribution guidelines.

## License

MIT OR Apache-2.0
