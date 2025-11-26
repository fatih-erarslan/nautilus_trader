# Polymarket Integration Tests Documentation

## Overview

This directory contains comprehensive integration tests for the Polymarket trading platform, including API integration, strategy execution, MCP server tools, performance benchmarks, and GPU acceleration validation.

## Test Structure

```
integration/
├── test_api_integration.py      # API client integration tests
├── test_strategy_integration.py # End-to-end strategy tests
├── test_mcp_integration.py      # MCP server tool tests
├── test_performance.py          # Performance benchmarks
├── test_gpu_acceleration.py     # GPU functionality tests
└── README.md                    # This documentation
```

## Running Integration Tests

### Prerequisites

1. Install test dependencies:
```bash
pip install -r src/polymarket/tests/requirements-test.txt
```

2. Set up environment variables:
```bash
export POLYMARKET_API_KEY="your_test_api_key"
export POLYMARKET_PRIVATE_KEY="your_test_private_key"
export POLYMARKET_ENV="testnet"  # Use testnet for integration tests
```

3. Ensure GPU drivers are installed (for GPU tests):
```bash
nvidia-smi  # Verify GPU availability
```

### Running All Integration Tests

```bash
# Run all integration tests with coverage
PYTHONPATH=/workspaces/ai-news-trader/src pytest src/polymarket/tests/integration/ \
    --cov=src/polymarket \
    --cov-report=html \
    --cov-report=term-missing \
    -v

# Run with parallel execution (faster)
PYTHONPATH=/workspaces/ai-news-trader/src pytest src/polymarket/tests/integration/ \
    -n auto \
    --cov=src/polymarket \
    -v
```

### Running Specific Test Categories

```bash
# API Integration Tests Only
pytest src/polymarket/tests/integration/test_api_integration.py -v

# Strategy Integration Tests
pytest src/polymarket/tests/integration/test_strategy_integration.py -v

# MCP Integration Tests
pytest src/polymarket/tests/integration/test_mcp_integration.py -v -m mcp

# Performance Tests
pytest src/polymarket/tests/integration/test_performance.py -v -m performance

# GPU Tests (requires CUDA)
pytest src/polymarket/tests/integration/test_gpu_acceleration.py -v -m gpu
```

### Test Markers

We use pytest markers to categorize tests:

- `@pytest.mark.integration` - General integration tests
- `@pytest.mark.mcp` - MCP server integration tests
- `@pytest.mark.performance` - Performance benchmark tests
- `@pytest.mark.gpu` - GPU-specific tests
- `@pytest.mark.benchmark` - Tests that include benchmarking

Run tests by marker:
```bash
# Run only MCP tests
pytest -m mcp

# Run performance and benchmark tests
pytest -m "performance or benchmark"

# Exclude GPU tests (if no GPU available)
pytest -m "not gpu"
```

## Test Coverage

### Achieving 100% Coverage

Our integration tests aim for 100% code coverage. To verify:

```bash
# Generate detailed coverage report
PYTHONPATH=/workspaces/ai-news-trader/src pytest src/polymarket/tests/ \
    --cov=src/polymarket \
    --cov-report=html \
    --cov-report=term-missing \
    --cov-fail-under=100

# View HTML report
open htmlcov/index.html
```

### Coverage Exclusions

Some code paths are excluded from coverage:
- Defensive error handlers that should never execute
- GPU fallback code when GPU is unavailable
- Platform-specific code branches

Add exclusions in `.coveragerc`:
```ini
[run]
omit = 
    */tests/*
    */test_*.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:
```

## Performance Testing

### Running Performance Benchmarks

```bash
# Run with benchmark plugin
pytest src/polymarket/tests/integration/test_performance.py \
    --benchmark-only \
    --benchmark-json=benchmark_results.json

# Compare benchmark results
pytest-benchmark compare benchmark_results.json

# Run stress tests
pytest src/polymarket/tests/integration/test_performance.py::test_high_frequency_trading_scenario -v
```

### Performance Targets

Our integration tests validate these performance targets:

| Metric | Target | Test |
|--------|--------|------|
| API Response Time | < 100ms avg | `test_api_response_time_benchmarks` |
| Market Analysis | > 20 markets/sec | `test_strategy_performance_benchmark` |
| WebSocket Messages | > 5 msg/sec/conn | `test_concurrent_websocket_connections` |
| GPU Speedup (Monte Carlo) | > 50x | `test_monte_carlo_gpu_acceleration` |
| Memory Growth | < 10% | `test_memory_usage_profiling` |

## MCP Integration Testing

### Testing MCP Tools

The MCP integration tests validate all 6 Polymarket tools:

1. **get_prediction_markets_tool** - Market discovery
2. **analyze_market_sentiment_tool** - Sentiment analysis with GPU
3. **get_market_orderbook_tool** - Orderbook data
4. **place_prediction_order_tool** - Order placement
5. **get_prediction_positions_tool** - Position tracking
6. **calculate_expected_value_tool** - EV calculation with Kelly Criterion

### Running MCP Server for Tests

```bash
# Start MCP server in test mode
python src/mcp/mcp_server_enhanced.py --test-mode

# Run MCP integration tests
pytest src/polymarket/tests/integration/test_mcp_integration.py -v
```

## GPU Acceleration Testing

### GPU Test Requirements

- NVIDIA GPU with CUDA support
- PyTorch with CUDA installed
- Minimum 4GB GPU memory

### GPU Test Categories

1. **Device Detection** - Verify GPU availability and properties
2. **Monte Carlo Simulations** - Test GPU speedup for simulations
3. **Neural Network Training** - Validate training performance
4. **Matrix Operations** - Benchmark linear algebra operations
5. **Memory Management** - Test allocation and cleanup

### Running GPU Tests

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Run GPU tests
pytest src/polymarket/tests/integration/test_gpu_acceleration.py -v -s

# Skip GPU tests if no GPU
pytest -m "not gpu"
```

## Continuous Integration

### GitHub Actions Workflow

Create `.github/workflows/polymarket-integration-tests.yml`:

```yaml
name: Polymarket Integration Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    paths:
      - 'src/polymarket/**'
      - 'tests/polymarket/**'

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r src/polymarket/tests/requirements-test.txt
    
    - name: Run integration tests
      env:
        POLYMARKET_API_KEY: ${{ secrets.TEST_API_KEY }}
        POLYMARKET_PRIVATE_KEY: ${{ secrets.TEST_PRIVATE_KEY }}
        PYTHONPATH: ${{ github.workspace }}/src
      run: |
        pytest src/polymarket/tests/integration/ \
          --cov=src/polymarket \
          --cov-report=xml \
          --junitxml=test-results.xml \
          -v
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
```

## Test Data Management

### Using Mock Data

The `test_utils.py` module provides comprehensive mock data generation:

```python
from polymarket.tests.test_utils import MockDataGenerator

# Initialize generator
generator = MockDataGenerator(seed=42)  # Use seed for reproducibility

# Generate test data
market = generator.generate_market()
orderbook = generator.generate_orderbook(market.id)
order = generator.generate_order(market.id)
signal = generator.generate_trading_signal(market.id)
```

### Test Fixtures

Common fixtures are available in `conftest.py`:

```python
@pytest.fixture
async def api_client():
    """Provides configured API client."""
    # Returns authenticated client

@pytest.fixture
def sample_markets():
    """Provides sample market data."""
    # Returns list of test markets

@pytest.fixture
async def websocket_server():
    """Provides mock WebSocket server."""
    # Returns running server instance
```

## Debugging Integration Tests

### Verbose Output

```bash
# Run with verbose output
pytest -vv -s

# Show local variables on failure
pytest -vv --showlocals

# Drop into debugger on failure
pytest --pdb
```

### Logging Configuration

Configure logging for tests in `pytest.ini`:

```ini
[tool:pytest]
log_cli = true
log_cli_level = INFO
log_cli_format = %(asctime)s [%(levelname)s] %(message)s
log_cli_date_format = %Y-%m-%d %H:%M:%S
```

### Performance Profiling

```bash
# Profile test execution
pytest --profile

# Generate call graph
pytest --profile-svg

# Memory profiling
pytest --memprof
```

## Best Practices

### 1. Test Isolation
- Each test should be independent
- Use fixtures for setup/teardown
- Mock external dependencies

### 2. Realistic Test Data
- Use the MockDataGenerator for consistent data
- Test edge cases and error conditions
- Include performance variations

### 3. Async Testing
- Use `pytest.mark.asyncio` for async tests
- Properly await all async operations
- Handle cleanup in finally blocks

### 4. Performance Testing
- Set realistic performance targets
- Test under various load conditions
- Monitor resource usage

### 5. Error Handling
- Test both success and failure paths
- Verify error messages and codes
- Test recovery mechanisms

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   export PYTHONPATH=/workspaces/ai-news-trader/src
   ```

2. **GPU Tests Failing**
   - Verify CUDA installation: `nvidia-smi`
   - Check PyTorch GPU support: `python -c "import torch; print(torch.cuda.is_available())"`

3. **Timeout Errors**
   - Increase test timeout: `pytest --timeout=300`
   - Check network connectivity
   - Verify API endpoints are accessible

4. **Coverage Not 100%**
   - Run with `--cov-report=term-missing` to see uncovered lines
   - Add tests for missing branches
   - Consider if exclusions are needed

### Getting Help

- Check test output for detailed error messages
- Review test logs in `test-results/`
- Run individual tests in isolation
- Use debugger for complex issues

## Contributing

When adding new integration tests:

1. Follow existing patterns and structure
2. Add appropriate test markers
3. Include docstrings explaining test purpose
4. Verify coverage remains at 100%
5. Update this documentation as needed

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest-asyncio](https://github.com/pytest-dev/pytest-asyncio)
- [Pytest-cov](https://pytest-cov.readthedocs.io/)
- [Pytest-benchmark](https://pytest-benchmark.readthedocs.io/)