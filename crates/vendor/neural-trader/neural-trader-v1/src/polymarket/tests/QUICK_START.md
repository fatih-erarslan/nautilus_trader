# Polymarket Integration Tests - Quick Start Guide

## Essential Commands

### 1. Run All Tests with Coverage
```bash
PYTHONPATH=/workspaces/ai-news-trader/src pytest src/polymarket/tests/ \
  --cov=src/polymarket \
  --cov-report=html \
  --cov-report=term-missing \
  -v
```

### 2. Run Specific Test Category
```bash
# API Integration Tests
pytest src/polymarket/tests/integration/test_api_integration.py -v

# Strategy Tests
pytest src/polymarket/tests/integration/test_strategy_integration.py -v

# MCP Tests
pytest src/polymarket/tests/integration/test_mcp_integration.py -v

# Performance Tests
pytest src/polymarket/tests/integration/test_performance.py -v

# GPU Tests (requires CUDA)
pytest src/polymarket/tests/integration/test_gpu_acceleration.py -v
```

### 3. Use Test Runner Script
```bash
# Run all tests with report
python src/polymarket/tests/run_integration_tests.py --report

# Run specific category
python src/polymarket/tests/run_integration_tests.py --category api_integration

# Run benchmarks
python src/polymarket/tests/run_integration_tests.py --category performance --benchmark

# Skip GPU tests
python src/polymarket/tests/run_integration_tests.py --no-gpu
```

### 4. View Coverage Report
```bash
# Generate HTML coverage report
pytest src/polymarket/tests/ --cov=src/polymarket --cov-report=html

# Open in browser
open htmlcov/index.html
```

### 5. Run Tests by Marker
```bash
# Only MCP tests
pytest -m mcp

# Only performance tests
pytest -m performance

# Exclude GPU tests
pytest -m "not gpu"

# Integration tests only
pytest -m integration
```

### 6. Check Coverage Threshold
```bash
# Ensure 100% coverage
pytest src/polymarket/tests/ --cov=src/polymarket --cov-fail-under=100
```

### 7. Run Tests in Parallel
```bash
# Use all CPU cores
pytest src/polymarket/tests/ -n auto

# Use specific number of workers
pytest src/polymarket/tests/ -n 4
```

### 8. Debug Failed Tests
```bash
# Show local variables
pytest -vv --showlocals

# Drop into debugger on failure
pytest --pdb

# Show full traceback
pytest --tb=long
```

### 9. Performance Profiling
```bash
# Profile test execution
pytest --profile

# Memory profiling
pytest --memprof
```

### 10. CI/CD Integration
```bash
# Generate XML report for CI
pytest --junit-xml=test-results.xml

# Generate coverage XML
pytest --cov=src/polymarket --cov-report=xml
```

## Environment Setup

Before running tests, ensure:

```bash
# Set Python path
export PYTHONPATH=/workspaces/ai-news-trader/src

# Set test environment
export POLYMARKET_ENV=test

# Install dependencies
pip install -r src/polymarket/tests/requirements-test.txt
```

## Test Structure

```
src/polymarket/tests/
├── integration/         # Integration tests
│   ├── test_api_integration.py
│   ├── test_strategy_integration.py
│   ├── test_mcp_integration.py
│   ├── test_performance.py
│   └── test_gpu_acceleration.py
├── unit/               # Unit tests
├── fixtures/           # Test fixtures
├── test_utils.py       # Test utilities
├── conftest.py         # Pytest configuration
├── pytest.ini          # Pytest settings
└── run_integration_tests.py  # Test runner
```

## Common Issues

1. **Import Errors**: Set PYTHONPATH correctly
2. **GPU Tests Fail**: Check CUDA installation with `nvidia-smi`
3. **Coverage Not 100%**: Run with `--cov-report=term-missing`
4. **Slow Tests**: Use `-n auto` for parallel execution

## Support

- View test documentation: `src/polymarket/tests/integration/README.md`
- Check example report: `src/polymarket/tests/example_coverage_report.html`
- Review summary: `INTEGRATION_TESTS_SUMMARY.md`