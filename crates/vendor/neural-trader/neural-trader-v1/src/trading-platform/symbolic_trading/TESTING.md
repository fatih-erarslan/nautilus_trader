# AI News Trading Platform - Testing Guide

## Overview

This document provides comprehensive testing instructions and best practices for the AI News Trading Platform. Our testing infrastructure follows Test-Driven Development (TDD) principles with a focus on quality, performance, and reliability.

## Testing Architecture

### Test Pyramid
Our testing strategy follows the test pyramid approach:
- **70% Unit Tests**: Fast, isolated tests for individual components
- **20% Integration Tests**: Component interaction and API integration tests  
- **10% End-to-End Tests**: Full system workflow tests

### Test Types
- **Unit Tests**: Test individual functions/classes in isolation
- **Integration Tests**: Test component interactions with mocked external services
- **End-to-End Tests**: Test complete user workflows
- **Performance Tests**: Load testing and benchmarking
- **Security Tests**: Vulnerability and penetration testing

## Quick Start

### Prerequisites
```bash
# Ensure Python 3.10+ is installed
python --version

# Install dependencies
pip install -r requirements.txt
```

### Running Tests

#### All Tests
```bash
# Run full test suite
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

#### Test Categories
```bash
# Unit tests only
pytest -m "unit or not (integration or e2e)"

# Integration tests
pytest -m integration

# End-to-end tests  
pytest -m e2e

# Performance tests
pytest -m performance --benchmark-only

# Security tests
pytest -m security
```

#### Parallel Execution
```bash
# Run tests in parallel (auto-detect CPU cores)
pytest -n auto

# Run with specific number of workers
pytest -n 4
```

## Test Structure

```
tests/
├── conftest.py              # Global test configuration
├── base.py                  # Base test classes
├── utils.py                 # Test utilities and helpers
├── factories/               # Test data factories
│   ├── __init__.py
│   ├── api_mocks.py        # External API mocks
│   ├── market_data.py      # Market data factories
│   ├── news.py             # News data factories
│   └── trading.py          # Trading data factories
├── fixtures/               # Test data fixtures
│   ├── market_data.json
│   ├── news_samples.json
│   └── api_responses.json
├── unit/                   # Unit tests
│   ├── test_*.py
│   └── */
├── integration/            # Integration tests
│   ├── test_*.py
│   └── */
├── e2e/                   # End-to-end tests
│   ├── test_*.py
│   └── */
└── performance/           # Performance tests
    ├── locustfile.py
    └── benchmarks/
```

## Writing Tests

### Base Test Classes

Choose the appropriate base class for your tests:

```python
from tests.base import BaseUnitTest, BaseIntegrationTest, BaseE2ETest

class TestMarketAnalysis(BaseUnitTest):
    """Unit tests for market analysis module."""
    
    def test_should_calculate_sentiment_when_given_news_text(self):
        # Arrange
        analyzer = SentimentAnalyzer()
        news_text = "Bitcoin reaches new all-time high"
        
        # Act
        result = analyzer.analyze(news_text)
        
        # Assert
        assert result.sentiment == "bullish"
        assert result.confidence > 0.8
```

### Using Test Factories

```python
from tests.factories import MarketDataFactory, NewsArticleFactory

class TestTradingSignals(BaseUnitTest):
    def test_should_generate_buy_signal_when_bullish_sentiment(self):
        # Arrange using factories
        market_data = MarketDataFactory(symbol="BTC_USDT", bullish=True)
        news = NewsArticleFactory(sentiment="bullish", strong_bullish=True)
        
        # Act & Assert
        signal = generate_trading_signal(market_data, news)
        assert signal.action == "buy"
```

### Mocking External Services

```python
from tests.factories import MockCryptoAPIFactory, MockLLMClientFactory

class TestMarketDataService(BaseIntegrationTest):
    def test_should_fetch_price_data_when_api_available(self):
        # Arrange
        mock_api = MockCryptoAPIFactory.create_mock()
        service = MarketDataService(api_client=mock_api)
        
        # Act
        price_data = service.get_current_price("BTC_USDT")
        
        # Assert
        assert price_data["symbol"] == "BTC_USDT"
        assert "price" in price_data
        mock_api.get_ticker.assert_called_once_with("BTC_USDT")
```

### Async Testing

```python
from tests.utils import AsyncTestHelper

class TestAsyncTrading(BaseUnitTest):
    async def test_should_execute_trade_async(self):
        # Arrange
        trader = AsyncTrader()
        
        # Act
        async with AsyncTestHelper.timeout_after(5.0):
            result = await trader.execute_trade("BTC_USDT", "buy", 0.1)
        
        # Assert
        assert result.status == "completed"
```

### Time-Based Testing

```python
from tests.utils import TimeHelper
from datetime import datetime, timezone

class TestTimeBasedFeatures(BaseUnitTest):
    def test_should_process_market_close_events(self):
        # Arrange
        market_close_time = datetime(2024, 1, 15, 21, 0, tzinfo=timezone.utc)
        
        # Act & Assert
        with TimeHelper.freeze_at(market_close_time):
            result = process_market_events()
            assert result.market_status == "closed"
```

## Test Configuration

### Environment Variables
```bash
# Test environment settings
export TESTING=true
export DATABASE_URL=sqlite:///:memory:
export REDIS_URL=redis://localhost:6379/1
export LOG_LEVEL=DEBUG

# API keys for testing (use test/mock keys)
export OPENROUTER_API_KEY=test_key
export NEWS_API_KEY=test_key
```

### Pytest Configuration

Key settings in `pytest.ini`:

```ini
[tool:pytest]
# Async support
asyncio_mode = auto

# Test discovery  
testpaths = tests
python_files = test_*.py

# Coverage settings
addopts = 
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
    -n auto  # Parallel execution

# Test markers
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests
    e2e: End-to-end tests
    slow: Tests that take > 1s
    security: Security tests
    performance: Performance benchmarks
```

## Performance Testing

### Load Testing with Locust

```bash
# Start basic load test
locust -f tests/performance/locustfile.py --host=http://localhost:8000

# Web UI load test
locust -f tests/performance/locustfile.py --host=http://localhost:8000 --web-host=0.0.0.0

# Headless load test
locust -f tests/performance/locustfile.py --host=http://localhost:8000 \
       --users 50 --spawn-rate 5 --run-time 5m --headless
```

### Benchmark Testing

```bash
# Run performance benchmarks
pytest -m benchmark --benchmark-only

# Generate benchmark report
pytest -m benchmark --benchmark-json=benchmark-results.json
```

### Performance Assertions

```python
from tests.utils import PerformanceHelper

class TestPerformance(BaseUnitTest):
    def test_should_process_market_data_quickly(self):
        # Arrange
        processor = MarketDataProcessor()
        data = large_market_dataset()
        
        # Act & Assert
        with PerformanceHelper.assert_max_time(1.0, "Market data processing"):
            result = processor.process(data)
```

## Security Testing

### Input Validation Tests

```python
from tests.base import BaseSecurityTest

class TestAPIValidation(BaseSecurityTest):
    def test_should_reject_sql_injection_attempts(self):
        """Test SQL injection protection."""
        sql_payloads = self.attack_vectors["sql_injection"]
        
        for payload in sql_payloads:
            with pytest.raises(ValidationError):
                api_endpoint_with_user_input(payload)
```

### Security Scanning

```bash
# Static security analysis
bandit -r src/ -f json -o security-report.json

# Dependency vulnerability scan
safety check --json --output vulnerability-report.json

# Secret detection
detect-secrets scan --all-files
```

## CI/CD Integration

### GitHub Actions

Our CI/CD pipeline automatically runs:

1. **Pre-commit checks**: Formatting, linting, type checking
2. **Unit tests**: Fast feedback on core functionality  
3. **Integration tests**: Component interaction validation
4. **Performance tests**: Benchmark regression detection
5. **Security tests**: Vulnerability scanning
6. **E2E tests**: Full workflow validation

### Quality Gates

Tests must pass these quality gates:

- **Test Coverage**: ≥80% overall, ≥95% for critical modules
- **Performance**: API responses <100ms, analysis <2s
- **Security**: Zero high-severity vulnerabilities
- **Reliability**: <5% flaky test rate

## Best Practices

### Test Writing Guidelines

1. **Follow AAA Pattern**: Arrange, Act, Assert
2. **One Assertion Per Test**: Clear failure identification
3. **Descriptive Names**: `test_should_X_when_Y`
4. **Independent Tests**: No shared state between tests
5. **Fast Tests**: Mock external dependencies
6. **Deterministic**: Same result every run

### Mock Usage

```python
# Good: Specific, meaningful mocks
mock_api = Mock()
mock_api.get_price.return_value = {"price": 50000, "symbol": "BTC_USDT"}

# Avoid: Overly broad mocks
mock_everything = Mock()
mock_everything.return_value = Mock()
```

### Factory Usage

```python
# Good: Use traits for variations
bullish_market = MarketDataFactory(bullish=True)
bearish_market = MarketDataFactory(bearish=True)

# Good: Override specific fields
custom_article = NewsArticleFactory(
    title="Custom headline",
    sentiment="neutral"
)
```

### Async Test Guidelines

```python
# Good: Proper async/await usage
async def test_async_operation(self):
    result = await async_function()
    assert result.success

# Good: Timeout protection
async def test_with_timeout(self):
    async with asyncio.timeout(5):
        await long_running_operation()
```

## Debugging Tests

### Running Individual Tests

```bash
# Run specific test file
pytest tests/unit/test_market_analysis.py

# Run specific test method
pytest tests/unit/test_market_analysis.py::TestSentimentAnalysis::test_bullish_sentiment

# Run with verbose output
pytest -v -s tests/unit/test_market_analysis.py
```

### Debugging Tools

```python
# Add breakpoints
import pdb; pdb.set_trace()

# Pretty print test data
import pprint
pprint.pprint(test_data)

# Capture logs
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Common Issues

1. **Flaky Tests**: Use proper mocking and time control
2. **Slow Tests**: Profile and optimize database/API calls
3. **Test Isolation**: Ensure proper cleanup in teardown methods
4. **Async Issues**: Use proper async test patterns

## Coverage Reporting

### Generate Reports

```bash
# HTML coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html

# Terminal coverage report
pytest --cov=src --cov-report=term-missing

# XML coverage for CI
pytest --cov=src --cov-report=xml
```

### Coverage Targets

- **Critical Modules**: 95% coverage (trading logic, risk management)
- **Core Modules**: 90% coverage (market data, news analysis)
- **Utility Modules**: 85% coverage (helpers, formatters)
- **Overall Target**: 80% minimum

## Maintenance

### Weekly Tasks

- Review and fix flaky tests
- Update test data and fixtures
- Check for new security vulnerabilities
- Performance regression analysis

### Monthly Tasks

- Refactor test code for better maintainability
- Update mock responses to match API changes
- Review and update test coverage targets
- Test infrastructure updates

### Release Tasks

- Full test suite execution
- Performance benchmark validation
- Security scan approval
- E2E test verification

## Troubleshooting

### Common Test Failures

| Error | Cause | Solution |
|-------|-------|----------|
| `ImportError` | Missing dependencies | Run `pip install -r requirements.txt` |
| `asyncio.TimeoutError` | Slow async operations | Increase timeout or mock external calls |
| `AssertionError` | Logic bugs or outdated tests | Debug and fix assertions |
| `ConnectionError` | Missing test services | Start required services (Redis, DB) |

### Getting Help

1. Check this documentation
2. Review existing test examples
3. Ask team members for code review
4. Use debugging tools and verbose output

## Resources

### Documentation
- [Pytest Documentation](https://docs.pytest.org/)
- [Factory Boy Guide](https://factoryboy.readthedocs.io/)
- [Locust Documentation](https://docs.locust.io/)

### Tools
- **IDE Extensions**: pytest runners, coverage visualization
- **Monitoring**: Test execution time tracking
- **Reporting**: Coverage reports, test analytics

### Training
- TDD workshops and pair programming sessions
- Code review best practices
- Testing strategy discussions

---

For questions or issues with testing, please contact the development team or create an issue in the project repository.