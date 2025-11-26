# Syndicate MCP Tools Test Suite

This directory contains comprehensive tests for the Syndicate Management System MCP (Model Context Protocol) integration.

## Overview

The test suite validates all 17 syndicate management tools across multiple dimensions:
- **Unit Tests**: Individual tool functionality
- **Integration Tests**: Interaction with trading systems
- **Security Tests**: Permission and access control
- **Performance Tests**: Concurrency and scalability
- **Stress Tests**: High-load scenarios

## Test Structure

### Core Test Files

1. **`test_syndicate_tools.py`** - Main test suite covering:
   - Member management operations (create, update, remove, list)
   - Fund allocation and distribution mechanisms
   - Voting system functionality
   - Analytics and reporting tools
   - Security and permission validation
   - Performance and concurrency handling

2. **`test_syndicate_integration.py`** - Integration tests for:
   - Syndicate-based trading workflows
   - Multi-syndicate coordination
   - Neural forecasting integration
   - Risk management integration
   - Real-time collaboration
   - Performance tracking

3. **`conftest.py`** - Shared fixtures and test configuration:
   - Mock MCP server
   - Test data generators
   - Performance monitoring utilities
   - Database and cache mocks

4. **`run_syndicate_tests.py`** - Test runner with options for:
   - Different test categories
   - Coverage reports
   - Stress testing
   - HTML report generation

## Running Tests

### Basic Usage

```bash
# Run all tests
python tests/mcp/run_syndicate_tests.py all

# Run only unit tests
python tests/mcp/run_syndicate_tests.py unit

# Run only integration tests
python tests/mcp/run_syndicate_tests.py integration

# Run security tests
python tests/mcp/run_syndicate_tests.py security

# Run performance tests
python tests/mcp/run_syndicate_tests.py performance
```

### Advanced Options

```bash
# Run with coverage report
python tests/mcp/run_syndicate_tests.py all --coverage

# Run specific test by name
python tests/mcp/run_syndicate_tests.py -k "test_create_member"

# Run stress tests
python tests/mcp/run_syndicate_tests.py stress --stress-duration=120 --concurrent-users=20

# Generate HTML report
python tests/mcp/run_syndicate_tests.py report

# Fail fast on first error
python tests/mcp/run_syndicate_tests.py all --failfast
```

### Using Pytest Directly

```bash
# Run all syndicate tests
pytest tests/mcp/test_syndicate_*.py -v

# Run with specific markers
pytest tests/mcp -m "unit and not slow" -v

# Run with coverage
pytest tests/mcp --cov=src.sports_betting.syndicate --cov-report=html

# Run specific test class
pytest tests/mcp/test_syndicate_tools.py::TestSyndicateMemberManagement -v
```

## Test Categories

### 1. Member Management Tests
- Creating new members with role assignment
- Updating member roles and permissions
- Removing members with fund settlement
- Listing and filtering members

### 2. Fund Management Tests
- Proportional fund allocation
- Performance-weighted distribution
- Contribution tracking over time
- Withdrawal processing and validation

### 3. Voting System Tests
- Proposal creation with validation
- Weighted voting mechanisms
- Vote tallying and outcome determination
- Proposal listing and filtering

### 4. Analytics Tests
- Performance metrics calculation
- Member analytics and rankings
- Bet history with filtering
- Risk metrics and reporting

### 5. Security Tests
- Permission verification
- SQL injection prevention
- Permission escalation prevention
- Audit log functionality

### 6. Integration Tests
- Complete betting workflows
- Multi-syndicate coordination
- Neural model integration
- Risk management integration
- Real-time collaboration

### 7. Performance Tests
- Concurrent vote casting
- Bulk member operations
- High-frequency fund allocations
- Stress testing under load

## Test Fixtures

### Mock Data Generators
- `mock_syndicate_data()`: Syndicate configuration
- `mock_members()`: Member profiles with different roles
- `mock_betting_opportunities()`: Sample betting scenarios
- `mock_voting_scenarios()`: Voting proposals

### Mock Services
- `mcp_server_mock`: Simulated MCP server
- `mock_database`: In-memory database
- `mock_redis`: Cache simulation
- `mock_websocket`: Real-time communication

## Coverage Requirements

The test suite aims for comprehensive coverage:
- **Minimum Coverage**: 80% for all modules
- **Critical Paths**: 95% coverage for security and fund management
- **Edge Cases**: Extensive testing of error conditions

## Performance Benchmarks

Expected performance metrics:
- **Member Operations**: < 100ms per operation
- **Fund Allocations**: < 200ms for complex calculations
- **Concurrent Users**: Support 50+ simultaneous users
- **Vote Processing**: < 50ms per vote
- **Analytics Generation**: < 1s for full reports

## Continuous Integration

The tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions configuration
- name: Run Syndicate Tests
  run: |
    python tests/mcp/run_syndicate_tests.py all --coverage --failfast
    
- name: Upload Coverage
  uses: codecov/codecov-action@v1
  with:
    file: ./htmlcov/coverage.xml
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure project root is in PYTHONPATH
   ```bash
   export PYTHONPATH=/workspaces/ai-news-trader:$PYTHONPATH
   ```

2. **Async Warnings**: Use pytest-asyncio for async test support
   ```bash
   pip install pytest-asyncio
   ```

3. **Coverage Missing**: Install pytest-cov
   ```bash
   pip install pytest-cov
   ```

### Debug Mode

Run tests with full output for debugging:
```bash
pytest tests/mcp/test_syndicate_tools.py -vvs --tb=long --log-cli-level=DEBUG
```

## Contributing

When adding new syndicate tools or features:

1. Add corresponding unit tests in `test_syndicate_tools.py`
2. Add integration tests in `test_syndicate_integration.py`
3. Update fixtures if new mock data is needed
4. Ensure all tests pass before submitting PR
5. Maintain minimum 80% coverage for new code

## Test Markers

Available pytest markers for selective test execution:
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.security` - Security tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.edge_case` - Edge case tests

## Future Enhancements

Planned improvements to the test suite:
- [ ] Property-based testing with Hypothesis
- [ ] Mutation testing for test quality
- [ ] Load testing with Locust
- [ ] Contract testing for API compatibility
- [ ] Snapshot testing for UI components