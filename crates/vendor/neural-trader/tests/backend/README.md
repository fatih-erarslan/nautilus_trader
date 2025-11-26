# Neural Trader Backend - Comprehensive Test Suite

Comprehensive test coverage for all 70+ functions and 7 classes in the Neural Trader backend.

## 📊 Test Coverage

**Target: 95%+ coverage across all metrics**

### Test Files

1. **unit-tests.test.js** - Unit tests for all functions
   - Trading functions (15+ tests)
   - Neural functions (20+ tests)
   - Sports betting (15+ tests)
   - Syndicate management (20+ tests)
   - Prediction markets (5+ tests)
   - E2B operations (10+ tests)
   - Security features (25+ tests)
   - News & analytics (10+ tests)
   - System functions (5+ tests)

2. **class-tests.test.js** - Class instance tests
   - FundAllocationEngine (20+ tests)
   - ProfitDistributionSystem (15+ tests)
   - WithdrawalManager (10+ tests)
   - MemberManager (25+ tests)
   - MemberPerformanceTracker (10+ tests)
   - VotingSystem (20+ tests)
   - CollaborationHub (15+ tests)

3. **integration-tests.test.js** - End-to-end workflows
   - Complete trading workflow
   - Syndicate lifecycle from creation to distribution
   - Swarm deployment and multi-agent coordination
   - Authentication and authorization flows
   - Neural model training to prediction pipeline
   - System health monitoring

4. **edge-cases.test.js** - Edge cases and error handling
   - Boundary conditions (numeric, string, array, date)
   - Invalid inputs (type mismatches, malformed JSON)
   - Security (SQL injection, XSS, path traversal)
   - Error scenarios (resource not found, file system errors)
   - Race conditions (concurrent operations)
   - Resource limits (memory, rate limits)
   - State consistency validation
   - Cleanup and recovery

5. **performance-tests.test.js** - Performance benchmarks
   - Execution time benchmarks for all operations
   - Throughput testing (sequential and batch)
   - Concurrent operation stress tests (50-200 concurrent)
   - Memory usage validation
   - Load testing (sustained, burst, stress)
   - Scalability tests
   - Benchmark summary reports

## 🚀 Running Tests

### Run All Tests
```bash
cd /workspaces/neural-trader
npm test -- tests/backend
```

### Run Specific Test Suite
```bash
# Unit tests only
npm test -- tests/backend/unit-tests.test.js

# Class tests only
npm test -- tests/backend/class-tests.test.js

# Integration tests only
npm test -- tests/backend/integration-tests.test.js

# Edge cases only
npm test -- tests/backend/edge-cases.test.js

# Performance tests only
npm test -- tests/backend/performance-tests.test.js
```

### Run with Coverage
```bash
npm test -- tests/backend --coverage
```

### Watch Mode (Development)
```bash
npm test -- tests/backend --watch
```

### Run Specific Test
```bash
npm test -- tests/backend/unit-tests.test.js -t "should list available strategies"
```

## 📈 Coverage Reports

After running tests with coverage, reports are generated in:
- `coverage/backend/lcov-report/index.html` - HTML report
- `coverage/backend/coverage-summary.json` - JSON summary
- Terminal output - Text summary

## 🧪 Test Structure

### Unit Tests
```javascript
describe('Function Group', () => {
  describe('specificFunction()', () => {
    it('should handle valid input', async () => {
      const result = await backend.specificFunction(params);
      expect(result).toHaveProperty('expectedField');
    });

    it('should reject invalid input', async () => {
      await expect(backend.specificFunction(invalid)).rejects.toThrow();
    });
  });
});
```

### Integration Tests
```javascript
describe('End-to-End Workflow', () => {
  it('should complete full workflow', async () => {
    // Step 1: Setup
    const resource = await backend.createResource();

    // Step 2: Operations
    const result = await backend.performOperations(resource);

    // Step 3: Verify
    expect(result.status).toBe('success');

    // Step 4: Cleanup
    await backend.cleanup(resource);
  });
});
```

### Performance Tests
```javascript
it('should complete operation within time limit', async () => {
  const start = Date.now();
  await backend.operation();
  const duration = Date.now() - start;

  expect(duration).toBeLessThan(1000); // 1 second
});
```

## 🎯 Test Categories

### 1. Trading Operations (125+ tests)
- Strategy listing and info
- Market analysis
- Trade simulation and execution
- Portfolio management
- Backtesting
- Risk analysis
- Optimization

### 2. Neural Operations (80+ tests)
- Forecasting
- Model training
- Evaluation
- Optimization
- Backtesting
- Model status

### 3. Sports Betting (75+ tests)
- Event listing
- Odds retrieval
- Arbitrage detection
- Kelly Criterion calculation
- Bet execution

### 4. Syndicate Management (150+ tests)
- Syndicate creation
- Member management
- Fund allocation
- Profit distribution
- Withdrawal management
- Voting system
- Collaboration

### 5. E2B Swarm (100+ tests)
- Swarm initialization
- Agent deployment
- Scaling
- Monitoring
- Performance tracking
- Health checks

### 6. Security (120+ tests)
- Authentication
- Authorization
- Rate limiting
- Input validation
- Audit logging
- CORS and security headers

### 7. Performance (50+ tests)
- Execution time
- Throughput
- Concurrent operations
- Memory usage
- Load testing
- Scalability

## 🔧 Configuration

### Jest Configuration
See `jest.config.js` for:
- Test environment setup
- Coverage thresholds (95%+)
- Test timeouts
- Reporter configuration

### Setup File
See `setup.js` for:
- Global test utilities
- Mock data generators
- Custom matchers
- Cleanup helpers

## 📝 Best Practices

1. **Descriptive Test Names**: Use clear, specific test descriptions
2. **Arrange-Act-Assert**: Follow AAA pattern
3. **Independent Tests**: Each test should be self-contained
4. **Cleanup**: Always cleanup resources after tests
5. **Async Handling**: Properly handle promises and async operations
6. **Error Cases**: Test both success and failure scenarios
7. **Edge Cases**: Test boundary conditions
8. **Performance**: Monitor execution times

## 🐛 Debugging Tests

### Enable Verbose Output
```bash
npm test -- tests/backend --verbose
```

### Debug Specific Test
```bash
node --inspect-brk node_modules/.bin/jest tests/backend/unit-tests.test.js
```

### Log Test Execution
```javascript
console.log('Current state:', JSON.stringify(result, null, 2));
```

## 📊 Coverage Goals

| Metric | Target | Current |
|--------|--------|---------|
| Statements | 95% | - |
| Branches | 95% | - |
| Functions | 95% | - |
| Lines | 95% | - |

## 🔄 Continuous Integration

Tests are run automatically on:
- Pull requests
- Main branch commits
- Release tags

CI Configuration: `.github/workflows/test.yml`

## 📚 Additional Resources

- [Jest Documentation](https://jestjs.io/docs/getting-started)
- [Testing Best Practices](https://github.com/goldbergyoni/javascript-testing-best-practices)
- [Neural Trader API Documentation](../../docs/API.md)

## 🤝 Contributing

When adding new features:
1. Write tests first (TDD)
2. Ensure 95%+ coverage
3. Add integration tests for workflows
4. Include edge cases
5. Add performance benchmarks
6. Update this README

## 📞 Support

For test-related issues:
- Check test output and error messages
- Review test setup in `setup.js`
- Verify backend installation
- Check for version compatibility

---

**Total Test Count**: 1000+ tests across all suites
**Estimated Runtime**: 5-10 minutes (full suite)
**Coverage Target**: 95%+ across all metrics
