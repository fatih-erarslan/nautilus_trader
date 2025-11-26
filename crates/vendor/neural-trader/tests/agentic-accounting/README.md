# Agentic Accounting System - Test Suite

Comprehensive testing infrastructure for the agentic accounting system following SPARC methodology and TDD practices.

## Quick Start

```bash
# Install dependencies
npm install

# Start test databases
npm run db:up

# Run all tests
npm test

# Run with coverage
npm run test:coverage
```

## Test Structure

```
tests/agentic-accounting/
├── unit/                          # Unit tests (70%)
│   ├── tax-calculations.test.ts   # Tax algorithm tests
│   ├── wash-sale.test.ts          # Wash sale detection
│   ├── fraud-detection.test.ts    # Forensic analysis
│   └── ...
├── integration/                   # Integration tests (20%)
│   ├── database-integration.test.ts
│   ├── agent-coordination.test.ts
│   ├── api-integration.test.ts
│   └── ...
├── e2e/                          # End-to-end tests (10%)
│   ├── full-workflow.test.ts     # Complete tax year
│   ├── tax-year-end.test.ts      # Year-end processing
│   └── compliance-audit.test.ts  # Compliance checks
├── fixtures/                     # Test data
│   ├── factories.ts              # Data factories
│   ├── sql/                      # SQL fixtures
│   └── json/                     # JSON fixtures
├── utils/                        # Test utilities
│   ├── test-setup.ts             # Global setup
│   ├── database-helpers.ts       # DB utilities
│   └── custom-matchers.ts        # Jest matchers
├── jest.config.js                # Jest configuration
├── vitest.config.ts              # Vitest configuration
├── docker-compose.test.yml       # Test databases
├── package.json                  # Dependencies
├── tsconfig.json                 # TypeScript config
└── README.md                     # This file
```

## Test Types

### Unit Tests

Fast, isolated tests for individual functions:

```bash
npm run test:unit
```

**Coverage**: >95%
**Speed**: <100ms per test

### Integration Tests

Multi-component tests with test database:

```bash
npm run test:integration
```

**Coverage**: >90%
**Speed**: <5s per test

### E2E Tests

Complete workflow tests:

```bash
npm run test:e2e
```

**Coverage**: Key workflows
**Speed**: <60s per test

## Available Scripts

```bash
# Test execution
npm test                  # Run all tests
npm run test:unit         # Unit tests only
npm run test:integration  # Integration tests
npm run test:e2e          # End-to-end tests
npm run test:watch        # Watch mode
npm run test:coverage     # With coverage report
npm run test:ci           # CI mode
npm run test:vitest       # Use Vitest
npm run test:vitest:ui    # Vitest UI

# Database management
npm run db:up             # Start test databases
npm run db:down           # Stop and remove databases
npm run db:logs           # View database logs
npm run db:reset          # Reset databases

# Code quality
npm run lint              # Lint tests
npm run typecheck         # Type check
```

## Test Databases

### Docker Compose Services

- **PostgreSQL 16 + pgvector**: Port 5433
- **Redis 7**: Port 6380
- **AgentDB**: In-memory

### Configuration

Default test database config:

```javascript
{
  postgres: {
    host: 'localhost',
    port: 5433,
    database: 'agentic_accounting_test',
    user: 'test_user',
    password: 'test_password'
  },
  redis: {
    host: 'localhost',
    port: 6380,
    password: 'test_redis_password'
  }
}
```

## Writing Tests

### Using Factories

```typescript
import { createTransaction, createLot, generateTransactions } from '../fixtures/factories';

// Single instance
const transaction = createTransaction({ asset: 'BTC' });

// Multiple instances
const transactions = generateTransactions(100);

// With overrides
const lot = createLot({
  quantity: '1.5',
  costBasis: '50000'
});
```

### Custom Matchers

```typescript
// Decimal comparison
expect(result.gain).toBeDecimal('50.123456');

// Close to (with precision)
expect(result.price).toBeDecimalCloseTo('50.123', 6);

// Range
expect(duration).toBeWithinRange(0, 100);

// Type checking
expect(tx).toHaveTransactionType('BUY');

// Validation
expect(tx.timestamp).toBeValidDate();
expect(tx.id).toBeValidUUID();
```

### Database Tests

```typescript
import { TestDatabaseLifecycle } from '../utils/database-helpers';

describe('Database Tests', () => {
  const dbLifecycle = new TestDatabaseLifecycle();

  beforeAll(() => dbLifecycle.setup());
  afterAll(() => dbLifecycle.teardown());
  beforeEach(() => dbLifecycle.cleanup());

  it('should persist data', async () => {
    const pool = dbLifecycle.getPool();
    // Use database...
  });
});
```

## Coverage Requirements

**Minimum thresholds: 90%**

- Statements: 90%
- Branches: 90%
- Functions: 90%
- Lines: 90%

View coverage report:

```bash
npm run test:coverage
open coverage/index.html
```

## Coordination Protocol

### Before Testing

```bash
npx claude-flow@alpha hooks pre-task --description "Test implementation"
npx claude-flow@alpha hooks session-restore --session-id "swarm-accounting-phase1"
```

### After Testing

```bash
npx claude-flow@alpha hooks post-task --task-id "test-setup"
npx claude-flow@alpha memory store \
  --key "swarm/tester/status" \
  --data '{"passed":145,"coverage":"92%"}'
```

## Best Practices

1. **Test First**: Write tests before implementation (TDD)
2. **One Assertion**: Each test should verify one behavior
3. **Descriptive Names**: Test names explain what and why
4. **Arrange-Act-Assert**: Clear test structure
5. **Mock Dependencies**: Keep tests isolated
6. **Test Edge Cases**: Boundary conditions and errors
7. **Fast Tests**: Unit tests run in <100ms
8. **Clean Up**: Always cleanup resources

## Troubleshooting

### Database Connection Errors

```bash
# Check containers
docker ps

# Restart databases
npm run db:reset

# View logs
npm run db:logs
```

### Test Timeouts

```typescript
// Increase timeout
it('long test', async () => {
  // test...
}, 60000); // 60 seconds
```

### Memory Issues

```bash
NODE_OPTIONS="--max-old-space-size=4096" npm test
```

## Documentation

- [Testing Guide](../../docs/agentic-accounting/TESTING-GUIDE.md)
- [Testing Strategy](/plans/agentic-accounting/refinement/01-testing-strategy.md)
- [Jest Documentation](https://jestjs.io/)
- [Vitest Documentation](https://vitest.dev/)

## Success Criteria

- ✅ Test framework configured
- ✅ Test databases working
- ✅ Fixtures and factories available
- ✅ Example tests written
- ✅ Coverage reporting configured
- ✅ Documentation complete
- ✅ 90%+ coverage threshold enforced

## Support

For questions or issues, check the [Testing Guide](../../docs/agentic-accounting/TESTING-GUIDE.md) or file an issue in the repository.
