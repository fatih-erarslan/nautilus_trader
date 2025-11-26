# Agentic Accounting System - Testing Guide

## Overview

This document describes the comprehensive testing strategy and best practices for the agentic accounting system.

## Table of Contents

- [Test Environment Setup](#test-environment-setup)
- [Testing Philosophy](#testing-philosophy)
- [Test Types](#test-types)
- [Running Tests](#running-tests)
- [Writing Tests](#writing-tests)
- [Best Practices](#best-practices)
- [Coordination Protocol](#coordination-protocol)
- [Coverage Requirements](#coverage-requirements)

## Test Environment Setup

### Prerequisites

- Node.js >= 18.0.0
- Docker and Docker Compose
- PostgreSQL 16 with pgvector
- Redis 7
- AgentDB

### Installation

```bash
# Navigate to test directory
cd tests/agentic-accounting

# Install dependencies
npm install

# Start test databases
npm run db:up

# Verify databases are running
npm run db:logs
```

### Environment Variables

Create a `.env.test` file:

```bash
# PostgreSQL
TEST_DB_HOST=localhost
TEST_DB_PORT=5433
TEST_DB_NAME=agentic_accounting_test
TEST_DB_USER=test_user
TEST_DB_PASSWORD=test_password

# Redis
TEST_REDIS_HOST=localhost
TEST_REDIS_PORT=6380
TEST_REDIS_PASSWORD=test_redis_password
TEST_REDIS_DB=0

# AgentDB
AGENTDB_PATH=:memory:
AGENTDB_DIMENSIONS=768
```

## Testing Philosophy

### Test-Driven Development (TDD)

We follow strict TDD practices:

1. **Write test first** - Define expected behavior
2. **Run test** - See it fail
3. **Write minimal code** - Make test pass
4. **Refactor** - Improve code quality
5. **Repeat** - Continue cycle

### Test Pyramid

```
         /\
        /E2E\      <- 10% (Slow, full workflows)
       /------\
      /Integr. \   <- 20% (Multi-component)
     /----------\
    /   Unit     \ <- 70% (Fast, isolated)
   /--------------\
```

## Test Types

### Unit Tests (70%)

**Location**: `tests/agentic-accounting/unit/`

**Purpose**: Test individual functions and modules in isolation

**Characteristics**:
- Fast (<100ms per test)
- No external dependencies
- High coverage (>95%)
- Deterministic

**Example**:

```typescript
describe('Tax Calculations', () => {
  it('should calculate FIFO disposal correctly', () => {
    const lots = [createLot({ quantity: '10', costBasis: '100' })];
    const sale = createSale({ quantity: '10', proceeds: '150' });

    const result = calculateFifo(sale, lots);

    expect(result.totalGain).toBeDecimal('50');
  });
});
```

### Integration Tests (20%)

**Location**: `tests/agentic-accounting/integration/`

**Purpose**: Test interaction between multiple components

**Characteristics**:
- Moderate speed (<5s per test)
- Uses test database
- Tests real integrations
- Cleanup between tests

**Example**:

```typescript
describe('Database Integration', () => {
  it('should persist and retrieve transactions', async () => {
    const transaction = createTransaction();
    await db.insert(transaction);

    const retrieved = await db.findById(transaction.id);
    expect(retrieved).toMatchObject(transaction);
  });
});
```

### End-to-End Tests (10%)

**Location**: `tests/agentic-accounting/e2e/`

**Purpose**: Test complete user workflows

**Characteristics**:
- Slow (<60s per test)
- Full system deployed
- Real-world scenarios
- Comprehensive validation

**Example**:

```typescript
describe('Tax Year Workflow', () => {
  it('should process full year and generate forms', async () => {
    // 1. Import transactions
    // 2. Calculate taxes
    // 3. Generate reports
    // 4. Verify audit trail

    expect(scheduleD).toContain('Schedule D');
  });
});
```

## Running Tests

### All Tests

```bash
npm test
```

### By Type

```bash
# Unit tests only
npm run test:unit

# Integration tests only
npm run test:integration

# E2E tests only
npm run test:e2e
```

### Watch Mode

```bash
npm run test:watch
```

### Coverage Report

```bash
npm run test:coverage

# Open HTML report
open coverage/index.html
```

### CI Mode

```bash
npm run test:ci
```

### Vitest (Alternative)

```bash
# Run all tests
npm run test:vitest

# Open UI
npm run test:vitest:ui
```

## Writing Tests

### Test Structure (Arrange-Act-Assert)

```typescript
describe('Feature', () => {
  it('should do something', () => {
    // Arrange - Set up test data
    const input = createTestData();

    // Act - Execute function under test
    const result = functionUnderTest(input);

    // Assert - Verify expectations
    expect(result).toBe(expected);
  });
});
```

### Using Factories

```typescript
import { createTransaction, createLot, generateTransactions } from '../fixtures/factories';

// Create single instance
const transaction = createTransaction({ asset: 'BTC' });

// Generate multiple instances
const transactions = generateTransactions(100, { type: 'BUY' });

// Create specific types
const sellTx = createSellTransaction({ quantity: '1.5' });
```

### Custom Matchers

```typescript
// Decimal comparison
expect(result.gain).toBeDecimal('50.123456');

// Decimal close to (with precision)
expect(result.price).toBeDecimalCloseTo('50.123', 6);

// Range checking
expect(duration).toBeWithinRange(0, 100);

// Transaction type
expect(tx).toHaveTransactionType('BUY');

// Date validation
expect(tx.timestamp).toBeValidDate();

// UUID validation
expect(tx.id).toBeValidUUID();
```

### Database Tests

```typescript
import { TestDatabaseLifecycle } from '../utils/database-helpers';

describe('Database Tests', () => {
  const dbLifecycle = new TestDatabaseLifecycle();

  beforeAll(async () => {
    await dbLifecycle.setup();
  });

  afterAll(async () => {
    await dbLifecycle.teardown();
  });

  beforeEach(async () => {
    await dbLifecycle.cleanup();
  });

  it('should work with database', async () => {
    const pool = dbLifecycle.getPool();
    const redis = dbLifecycle.getRedis();
    const agentdb = dbLifecycle.getAgentDB();

    // Use databases...
  });
});
```

### Performance Tests

```typescript
import { measureTime } from '../utils/test-setup';

it('should complete in under 100ms', async () => {
  const { result, duration } = await measureTime(async () => {
    return await expensiveOperation();
  });

  expect(duration).toBeLessThan(100);
  expect(result).toBeDefined();
});
```

## Best Practices

### 1. Test Independence

Each test should be completely independent:

```typescript
// ✅ Good - Independent
it('test 1', () => {
  const data = createTestData();
  // test...
});

it('test 2', () => {
  const data = createTestData();
  // test...
});

// ❌ Bad - Shared state
let sharedData;

it('test 1', () => {
  sharedData = createTestData();
});

it('test 2', () => {
  // Depends on test 1
  expect(sharedData).toBeDefined();
});
```

### 2. Clear Test Names

```typescript
// ✅ Good - Descriptive
it('should calculate FIFO with multiple lots correctly', () => {});
it('should throw InsufficientQuantityError when quantity exceeds available', () => {});

// ❌ Bad - Vague
it('works', () => {});
it('test 1', () => {});
```

### 3. One Assertion Per Test

```typescript
// ✅ Good - Single responsibility
it('should return correct total gain', () => {
  expect(result.totalGain).toBeDecimal('50');
});

it('should return correct number of disposals', () => {
  expect(result.disposals).toHaveLength(2);
});

// ❌ Bad - Multiple assertions
it('should calculate correctly', () => {
  expect(result.totalGain).toBeDecimal('50');
  expect(result.disposals).toHaveLength(2);
  expect(result.method).toBe('FIFO');
  expect(result.timestamp).toBeDefined();
});
```

### 4. Mock External Dependencies

```typescript
// ✅ Good - Mocked
jest.mock('external-api');

it('should handle API response', async () => {
  externalApi.fetch.mockResolvedValue(mockData);

  const result = await service.getData();
  expect(result).toBeDefined();
});

// ❌ Bad - Real API call
it('should get real data', async () => {
  const result = await service.getData(); // Slow, unreliable
  expect(result).toBeDefined();
});
```

### 5. Test Edge Cases

```typescript
describe('Edge Cases', () => {
  it('should handle empty input', () => {
    expect(process([])).toEqual([]);
  });

  it('should handle null input', () => {
    expect(process(null)).toEqual(null);
  });

  it('should handle maximum values', () => {
    const max = Number.MAX_SAFE_INTEGER;
    expect(process(max)).toBeDefined();
  });

  it('should handle decimal precision', () => {
    const result = calculate('0.00000001');
    expect(result).toBeDecimal('0.00000001');
  });
});
```

### 6. Cleanup Resources

```typescript
describe('Resource Tests', () => {
  let connection;

  beforeEach(async () => {
    connection = await createConnection();
  });

  afterEach(async () => {
    // Always cleanup!
    await connection.close();
  });

  it('should use connection', async () => {
    // test...
  });
});
```

## Coordination Protocol

### Before Testing

```bash
# Run pre-task hook
npx claude-flow@alpha hooks pre-task --description "Test implementation"

# Restore session if needed
npx claude-flow@alpha hooks session-restore --session-id "swarm-accounting-phase1"
```

### During Testing

```bash
# Notify progress
npx claude-flow@alpha hooks notify --message "Completed unit tests"

# Store patterns
npx claude-flow@alpha memory store \
  --key "swarm/tester/patterns" \
  --data '{"pattern":"factory-pattern","status":"implemented"}'
```

### After Testing

```bash
# Post-task hook
npx claude-flow@alpha hooks post-task --task-id "test-setup"

# Store test results
npx claude-flow@alpha memory store \
  --key "swarm/shared/test-results" \
  --data '{"passed":145,"failed":0,"coverage":"92%"}'
```

## Coverage Requirements

### Thresholds

All packages must meet these minimum coverage thresholds:

- **Statements**: 90%
- **Branches**: 90%
- **Functions**: 90%
- **Lines**: 90%

### Viewing Coverage

```bash
# Generate coverage report
npm run test:coverage

# Open HTML report
open coverage/index.html
```

### Coverage Report Structure

```
coverage/
├── index.html          # Main coverage report
├── lcov.info          # LCOV format
├── coverage-final.json # JSON format
└── [package-name]/    # Per-package reports
```

### CI Integration

Coverage is automatically checked in CI:

```yaml
- name: Test with Coverage
  run: npm run test:ci

- name: Check Coverage Thresholds
  run: |
    if [ $(jq '.total.statements.pct < 90' coverage/coverage-summary.json) ]; then
      echo "Coverage below 90%"
      exit 1
    fi
```

## Troubleshooting

### Database Connection Errors

```bash
# Check if containers are running
docker ps

# Restart databases
npm run db:reset

# View logs
npm run db:logs
```

### Test Timeouts

```typescript
// Increase timeout for specific test
it('long running test', async () => {
  // test...
}, 60000); // 60 seconds

// Or in describe block
describe('Slow Tests', () => {
  jest.setTimeout(60000);

  it('test', async () => {
    // test...
  });
});
```

### Memory Issues

```bash
# Run with increased memory
NODE_OPTIONS="--max-old-space-size=4096" npm test
```

## Additional Resources

- [Jest Documentation](https://jestjs.io/docs/getting-started)
- [Vitest Documentation](https://vitest.dev/guide/)
- [Testing Best Practices](https://testingjavascript.com/)
- [SPARC Testing Strategy](/plans/agentic-accounting/refinement/01-testing-strategy.md)

## Support

For questions or issues:

1. Check this guide
2. Review test examples in `/tests/agentic-accounting/`
3. Consult the team via coordination memory
4. File an issue in the repository

---

**Remember**: Good tests are an investment in code quality and maintainability. Write tests that you'd want to debug in 6 months!
