# Troubleshooting Guide

Common issues and solutions for Neural Trader examples.

## Table of Contents

- [Installation Issues](#installation-issues)
- [AgentDB Issues](#agentdb-issues)
- [Swarm Coordination Issues](#swarm-coordination-issues)
- [Performance Issues](#performance-issues)
- [OpenRouter Integration Issues](#openrouter-integration-issues)
- [Build and TypeScript Issues](#build-and-typescript-issues)
- [Memory Issues](#memory-issues)
- [Testing Issues](#testing-issues)

---

## Installation Issues

### Issue: `npm install` fails with EACCES error

**Symptom**:
```
npm ERR! code EACCES
npm ERR! syscall mkdir
npm ERR! path /usr/local/lib/node_modules
npm ERR! errno -13
```

**Solution**:
```bash
# Option 1: Use npx (recommended)
npx @neural-trader/example-market-microstructure

# Option 2: Fix npm permissions
sudo chown -R $(whoami) ~/.npm
sudo chown -R $(whoami) /usr/local/lib/node_modules

# Option 3: Use nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 18
nvm use 18
```

### Issue: Workspace dependencies not resolving

**Symptom**:
```
Error: Cannot find module '@neural-trader/predictor'
```

**Solution**:
```bash
# From repository root
npm install

# Rebuild all workspaces
npm run build --workspaces

# Or build specific example
cd packages/examples/market-microstructure
npm install
npm run build
```

### Issue: Peer dependency conflicts

**Symptom**:
```
npm ERR! ERESOLVE unable to resolve dependency tree
npm ERR! Found: agentdb@2.0.0
npm ERR! Could not resolve dependency: peer agentdb@"^1.0.0"
```

**Solution**:
```bash
# Use --legacy-peer-deps
npm install --legacy-peer-deps

# Or update package.json to use compatible versions
{
  "dependencies": {
    "agentdb": "^2.0.0"  // Update to latest
  }
}
```

---

## AgentDB Issues

### Issue: `AgentDB initialization failed`

**Symptom**:
```
Error: AgentDB initialization failed: EACCES: permission denied
```

**Solution**:
```bash
# Ensure write permissions in current directory
chmod 755 ./

# Or specify a writable path
const learner = new SelfLearningOptimizer('/tmp/memory.db');
```

### Issue: Database file locked

**Symptom**:
```
Error: database is locked
```

**Solution**:
```typescript
// Ensure proper cleanup
try {
  await component.initialize();
  await component.run();
} finally {
  await component.close(); // Always close connections
}

// Or remove stale lock file
// rm -f ./memory.db-wal ./memory.db-shm
```

### Issue: Out of memory with large datasets

**Symptom**:
```
FATAL ERROR: Reached heap limit Allocation failed - JavaScript heap out of memory
```

**Solution**:
```typescript
// Enable quantization for memory reduction
const learner = new SelfLearningOptimizer('./memory.db', {
  quantization: '8bit' // or '4bit' for even more savings
});

// Limit in-memory history
class Component {
  private maxHistory = 1000; // Reduce this value

  addData(data: any) {
    this.history.push(data);
    if (this.history.length > this.maxHistory) {
      this.history = this.history.slice(-this.maxHistory);
    }
  }
}

// Increase Node.js heap size
// node --max-old-space-size=4096 your-script.js
```

### Issue: Slow similarity search

**Symptom**:
Similarity search taking seconds instead of milliseconds.

**Solution**:
```typescript
// Enable HNSW indexing for 150x speedup
await agentdb.initialize({
  indexType: 'hnsw',
  M: 16,              // Increase for better recall
  efConstruction: 200 // Increase for better index quality
});

// Build index after batch inserts
await agentdb.buildIndex({
  collection: 'patterns',
  indexType: 'hnsw'
});
```

### Issue: `RL algorithm not found`

**Symptom**:
```
Error: Unknown RL algorithm: 'decision_transformer'
```

**Solution**:
```bash
# Update agentdb to latest version
npm install agentdb@latest

# Check available algorithms
import { AgentDB } from 'agentdb';
const db = new AgentDB('./memory.db');
console.log(db.getSupportedRLAlgorithms());
```

---

## Swarm Coordination Issues

### Issue: Swarm agents not making progress

**Symptom**:
Swarm runs for many generations without improvement.

**Solution**:
```typescript
// Increase population diversity
const swarm = new SwarmFeatureEngineer({
  numAgents: 50,          // Increase population size
  mutationRate: 0.3,      // Increase mutation for more exploration
  crossoverRate: 0.7,
  eliteSize: 5            // Keep more elite agents
});

// Add early stopping
async optimize(): Promise<Solution> {
  let noImprovementCount = 0;
  const noImprovementLimit = 10;

  for (let gen = 0; gen < this.maxGenerations; gen++) {
    const currentBest = this.getBestAgent();

    if (currentBest.fitness <= this.previousBest) {
      noImprovementCount++;
    } else {
      noImprovementCount = 0;
      this.previousBest = currentBest.fitness;
    }

    if (noImprovementCount >= noImprovementLimit) {
      console.log(`Early stopping at generation ${gen}`);
      break;
    }

    await this.evolve();
  }

  return this.getBestAgent();
}
```

### Issue: Swarm consuming too much memory

**Symptom**:
```
FATAL ERROR: JavaScript heap out of memory
```

**Solution**:
```typescript
// Reduce swarm size
const swarm = new SwarmFeatureEngineer({
  numAgents: 20,        // Reduce from 50
  generations: 30       // Reduce from 100
});

// Process agents in batches
async evaluatePopulation(agents: Agent[]): Promise<void> {
  const batchSize = 10;

  for (let i = 0; i < agents.length; i += batchSize) {
    const batch = agents.slice(i, i + batchSize);
    await Promise.all(batch.map(a => this.evaluateFitness(a)));

    // Force garbage collection between batches
    if (global.gc) global.gc();
  }
}
```

### Issue: Swarm taking too long to complete

**Symptom**:
Swarm optimization taking minutes instead of seconds.

**Solution**:
```typescript
// Reduce complexity
const swarm = new SwarmFeatureEngineer({
  numAgents: 20,      // Fewer agents
  generations: 20,    // Fewer generations
  useOpenRouter: false // Disable LLM calls for faster execution
});

// Parallelize evaluation
async evaluatePopulation(agents: Agent[]): Promise<void> {
  // Use Promise.all for parallel execution
  await Promise.all(
    agents.map(agent => this.evaluateFitness(agent))
  );
}

// Cache expensive computations
private fitnessCache = new Map<string, number>();

async evaluateFitness(agent: Agent): Promise<number> {
  const key = this.agentToKey(agent);

  if (this.fitnessCache.has(key)) {
    return this.fitnessCache.get(key)!;
  }

  const fitness = await this.computeFitness(agent);
  this.fitnessCache.set(key, fitness);
  return fitness;
}
```

---

## Performance Issues

### Issue: Slow order book analysis

**Symptom**:
Order book analysis taking >10ms per snapshot.

**Solution**:
```typescript
// Pre-compute and cache values
class OrderBookAnalyzer {
  private previousMetrics?: Metrics;

  analyzeOrderBook(orderBook: OrderBook): Metrics {
    // Use incremental updates if order book changed slightly
    if (this.previousMetrics && this.hasSmallChange(orderBook)) {
      return this.incrementalUpdate(orderBook, this.previousMetrics);
    }

    // Full computation
    const metrics = this.computeMetrics(orderBook);
    this.previousMetrics = metrics;
    return metrics;
  }
}

// Limit history size
private maxHistory = 100; // Reduce from 1000

// Use typed arrays for numeric data
private prices = new Float64Array(100);
private sizes = new Float64Array(100);
```

### Issue: Portfolio optimization taking too long

**Symptom**:
Mean-Variance optimization taking >1 second.

**Solution**:
```typescript
// Reduce iterations
const optimizer = new MeanVarianceOptimizer(assets, correlationMatrix, {
  maxIterations: 100,  // Reduce from 1000
  tolerance: 0.001     // Increase tolerance
});

// Use warm start
const result = optimizer.optimize({
  minWeight: 0.05,
  maxWeight: 0.40,
  targetReturn: 0.14,
  initialWeights: previousWeights // Warm start with previous solution
});

// Simplify correlation matrix
function simplifyCorrelationMatrix(matrix: number[][]): number[][] {
  // Keep only top correlations, set others to 0
  return matrix.map(row =>
    row.map(val => Math.abs(val) > 0.3 ? val : 0)
  );
}
```

### Issue: High CPU usage

**Symptom**:
Node.js process using 100% CPU continuously.

**Solution**:
```typescript
// Add delays between intensive operations
async function processWithBackpressure<T>(
  items: T[],
  processor: (item: T) => Promise<void>
): Promise<void> {
  for (const item of items) {
    await processor(item);

    // Yield to event loop every 10 items
    if (items.indexOf(item) % 10 === 0) {
      await new Promise(resolve => setImmediate(resolve));
    }
  }
}

// Throttle swarm evaluations
async evaluatePopulation(agents: Agent[]): Promise<void> {
  for (let i = 0; i < agents.length; i++) {
    await this.evaluateFitness(agents[i]);

    // Small delay to prevent CPU saturation
    if (i % 10 === 0) {
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }
}
```

---

## OpenRouter Integration Issues

### Issue: `OpenRouter API key not set`

**Symptom**:
```
Error: OPENROUTER_API_KEY not set
```

**Solution**:
```bash
# Set environment variable
export OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Or in code
process.env.OPENROUTER_API_KEY = 'sk-or-v1-your-key-here';

# Or disable OpenRouter
const swarm = new SwarmFeatureEngineer({
  useOpenRouter: false
});
```

### Issue: OpenRouter rate limit exceeded

**Symptom**:
```
Error: Rate limit exceeded: 429 Too Many Requests
```

**Solution**:
```typescript
// Implement rate limiting
class RateLimiter {
  private requests: number[] = [];

  async waitIfNeeded(): Promise<void> {
    const now = Date.now();
    this.requests = this.requests.filter(t => now - t < 60000);

    if (this.requests.length >= 20) { // 20 requests per minute
      const waitTime = 60000 - (now - this.requests[0]);
      await new Promise(resolve => setTimeout(resolve, waitTime));
    }

    this.requests.push(now);
  }
}

// Reduce OpenRouter usage
const swarm = new SwarmFeatureEngineer({
  useOpenRouter: true,
  numAgents: 10,          // Fewer agents = fewer API calls
  generations: 20
});
```

### Issue: OpenRouter timeout

**Symptom**:
```
Error: Request timeout after 30000ms
```

**Solution**:
```typescript
// Increase timeout
const swarm = new SwarmFeatureEngineer({
  useOpenRouter: true,
  openRouterConfig: {
    timeout: 60000  // 60 seconds
  }
});

// Add retry logic
async function callOpenRouterWithRetry(
  prompt: string,
  maxRetries = 3
): Promise<string> {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await this.callOpenRouter(prompt);
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      console.warn(`Attempt ${i + 1} failed, retrying...`);
      await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
    }
  }
}
```

---

## Build and TypeScript Issues

### Issue: TypeScript compilation errors

**Symptom**:
```
error TS2307: Cannot find module '@neural-trader/predictor'
```

**Solution**:
```bash
# Ensure all dependencies are installed
npm install

# Build dependencies first
cd packages/predictor && npm run build
cd ../core && npm run build

# Then build your example
cd packages/examples/market-microstructure
npm run build

# Or use workspaces from root
npm run build --workspaces
```

### Issue: Module resolution errors

**Symptom**:
```
Cannot find module '@neural-trader/predictor' or its corresponding type declarations
```

**Solution**:
```json
// tsconfig.json - Add paths
{
  "compilerOptions": {
    "paths": {
      "@neural-trader/predictor": ["../../../predictor/src"],
      "@neural-trader/core": ["../../../core/src"]
    }
  }
}

// Or use workspace protocol in package.json
{
  "dependencies": {
    "@neural-trader/predictor": "workspace:*"
  }
}
```

### Issue: Jest ESM module errors

**Symptom**:
```
SyntaxError: Cannot use import statement outside a module
```

**Solution**:
```javascript
// jest.config.js
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  extensionsToTreatAsEsm: ['.ts'],
  moduleNameMapper: {
    '^(\\.{1,2}/.*)\\.js$': '$1',
  },
  transform: {
    '^.+\\.tsx?$': [
      'ts-jest',
      {
        useESM: true,
      },
    ],
  },
};

// package.json
{
  "type": "module",
  "scripts": {
    "test": "node --experimental-vm-modules node_modules/jest/bin/jest.js"
  }
}
```

---

## Memory Issues

### Issue: Memory leak in long-running process

**Symptom**:
Memory usage continuously increasing over time.

**Solution**:
```typescript
// Clear caches periodically
class Component {
  private cache = new Map<string, any>();

  private clearCacheIfNeeded(): void {
    if (this.cache.size > 1000) {
      // Keep only most recent 500 entries
      const entries = Array.from(this.cache.entries());
      this.cache.clear();

      entries.slice(-500).forEach(([key, value]) => {
        this.cache.set(key, value);
      });
    }
  }

  // Call periodically
  private setupMaintenanceTimer(): void {
    setInterval(() => {
      this.clearCacheIfNeeded();

      if (global.gc) {
        global.gc(); // Manual garbage collection
      }
    }, 60000); // Every minute
  }
}

// Run with GC enabled
// node --expose-gc your-script.js
```

### Issue: Large arrays consuming memory

**Symptom**:
Memory spikes when processing large datasets.

**Solution**:
```typescript
// Use streams instead of loading all data
import { createReadStream } from 'fs';
import { parse } from 'csv-parse';

async function* processLargeFile(filePath: string) {
  const stream = createReadStream(filePath)
    .pipe(parse({ columns: true }));

  for await (const row of stream) {
    yield processRow(row);
  }
}

// Process in batches
async function processBatch(items: any[]): Promise<void> {
  const batchSize = 100;

  for (let i = 0; i < items.length; i += batchSize) {
    const batch = items.slice(i, i + batchSize);
    await processItems(batch);

    // Clear processed items
    batch.length = 0;

    // Force GC if available
    if (global.gc) global.gc();
  }
}
```

---

## Testing Issues

### Issue: Tests timing out

**Symptom**:
```
Error: Timeout - Async callback was not invoked within the 5000 ms timeout
```

**Solution**:
```typescript
// Increase timeout for specific test
it('runs long optimization', async () => {
  const result = await optimizer.optimize(complexData);
  expect(result).toBeDefined();
}, 30000); // 30 second timeout

// Or globally in jest.config.js
module.exports = {
  testTimeout: 30000
};
```

### Issue: Flaky tests

**Symptom**:
Tests passing sometimes, failing other times.

**Solution**:
```typescript
// Use fixed random seed
beforeAll(() => {
  Math.random = () => 0.5; // Deterministic
});

// Increase tolerance for floating point
expect(result).toBeCloseTo(0.123, 2); // 2 decimal places

// Add retries for external dependencies
jest.retryTimes(3);

// Mock external APIs
jest.mock('openai', () => ({
  OpenAI: jest.fn().mockImplementation(() => ({
    chat: {
      completions: {
        create: jest.fn().mockResolvedValue({
          choices: [{ message: { content: 'Mocked' } }]
        })
      }
    }
  }))
}));
```

### Issue: Cannot cleanup test resources

**Symptom**:
```
Error: ENOENT: no such file or directory
```

**Solution**:
```typescript
// Proper cleanup in tests
import { rm } from 'fs/promises';

describe('Component', () => {
  const testDbPath = './test-memory.db';
  let component: Component;

  beforeEach(async () => {
    component = new Component(testDbPath);
    await component.initialize();
  });

  afterEach(async () => {
    // Close connections first
    await component.close();

    // Then cleanup files
    try {
      await rm(testDbPath, { recursive: true, force: true });
      await rm(`${testDbPath}-wal`, { force: true });
      await rm(`${testDbPath}-shm`, { force: true });
    } catch (error) {
      // Ignore cleanup errors
    }
  });
});
```

---

## Getting Help

If you encounter issues not covered here:

1. **Check Documentation**: Review [Architecture](./ARCHITECTURE.md) and [Best Practices](./BEST_PRACTICES.md)
2. **Search Issues**: Check [GitHub Issues](https://github.com/ruvnet/neural-trader/issues)
3. **Enable Debug Logging**: Set `DEBUG=neural-trader:*` environment variable
4. **Create Minimal Reproduction**: Isolate the issue to minimal code
5. **Open Issue**: Report with reproduction steps and environment details

---

## Debug Mode

Enable detailed logging:

```typescript
// Set environment variable
process.env.DEBUG = 'neural-trader:*';

// Or in code
const DEBUG = true;

if (DEBUG) {
  console.log('Debug info:', data);
}
```

---

Built with ❤️ by the Neural Trader team
