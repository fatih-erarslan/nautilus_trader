# Neural Forecasting Test Suite

Comprehensive test suite for neural forecasting components in the AI News Trading Platform, providing 100% test coverage with performance benchmarking and GPU compatibility testing.

## 🎯 Overview

This test suite validates the neural forecasting implementation including:

- **NHITS Model Integration** - Core neural network components
- **MCP Tool Integration** - Model Control Protocol server tools
- **Real-Time Inference** - Low-latency prediction engine
- **Multi-Asset Processing** - Parallel forecasting capabilities
- **GPU Acceleration** - CUDA optimization and compatibility
- **Performance Benchmarking** - Latency, throughput, and memory efficiency
- **Data Pipeline** - Preprocessing and validation components

## 📁 Test Structure

```
tests/neural/
├── unit/                      # Unit tests for individual components
│   ├── test_nhits_integration.py      # Core NHITS model tests
│   ├── test_data_processing.py        # Data pipeline tests
│   ├── test_model_components.py       # Model component tests
│   └── test_inference_engine.py       # Inference engine tests
├── integration/               # Integration tests
│   ├── test_mcp_integration.py        # MCP server integration
│   ├── test_trading_platform.py       # Trading platform integration
│   └── test_data_pipeline.py          # End-to-end data flow
├── performance/               # Performance tests
│   ├── test_inference_performance.py  # Latency & throughput tests
│   ├── test_training_performance.py   # Training performance tests
│   ├── test_memory_performance.py     # Memory efficiency tests
│   └── test_scalability.py           # Scalability tests
├── utils/                     # Test utilities
│   ├── fixtures.py                   # Pytest fixtures
│   ├── data_generators.py            # Test data generation
│   ├── gpu_utils.py                  # GPU testing utilities
│   ├── mock_objects.py               # Mock implementations
│   └── performance_utils.py          # Performance testing tools
├── conftest.py                # Pytest configuration
├── pytest.ini                # Test configuration
├── run_tests.py              # Test runner script
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements-neural.txt

# For GPU testing (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Running Tests

```bash
# Run all tests
python tests/neural/run_tests.py --all

# Run specific test categories
python tests/neural/run_tests.py --unit
python tests/neural/run_tests.py --integration
python tests/neural/run_tests.py --performance

# Run GPU tests (if GPU available)
python tests/neural/run_tests.py --gpu

# Quick mode (skip slow tests)
python tests/neural/run_tests.py --all --quick

# Generate coverage report
python tests/neural/run_tests.py --coverage
```

### Using PyTest Directly

```bash
# Run all neural tests
pytest tests/neural/ -v

# Run specific test files
pytest tests/neural/unit/test_nhits_integration.py -v

# Run tests with coverage
pytest tests/neural/ --cov=src --cov-report=html

# Run only GPU tests
pytest tests/neural/ -m gpu

# Run performance tests
pytest tests/neural/ -m performance
```

## 🧪 Test Categories

### Unit Tests (`-m unit`)

Test individual components in isolation:

- ✅ **NHITS Model Tests** - Model creation, forward pass, serialization
- ✅ **Data Processing Tests** - Time series generation, format conversion
- ✅ **Model Components Tests** - Individual neural network layers
- ✅ **Inference Engine Tests** - Real-time prediction engine

### Integration Tests (`-m integration`)

Test component interactions:

- ✅ **MCP Integration** - Neural tools through MCP server
- ✅ **Trading Platform Integration** - End-to-end trading workflows
- ✅ **Data Pipeline Integration** - Complete data flow validation

### Performance Tests (`-m performance`)

Benchmark performance characteristics:

- ⚡ **Inference Latency** - Sub-50ms prediction requirements
- 📈 **Throughput Tests** - 1000+ predictions per second
- 💾 **Memory Efficiency** - Memory usage optimization
- 📊 **Scalability Tests** - Performance scaling analysis

### GPU Tests (`-m gpu`)

Validate GPU acceleration:

- 🖥️ **GPU Compatibility** - CUDA functionality validation
- ⚡ **GPU Performance** - Acceleration benchmarking
- 💾 **GPU Memory Management** - Memory leak detection
- 🔄 **CPU/GPU Comparison** - Performance comparison tests

## 📊 Performance Targets

The test suite validates these performance requirements:

| Metric | Target | Test Category |
|--------|--------|---------------|
| **Inference Latency** | < 50ms (P99) | Performance |
| **Throughput** | > 1,000 predictions/sec | Performance |
| **Memory Usage** | < 512MB peak | Performance |
| **GPU Speedup** | > 2x vs CPU | GPU |
| **Accuracy (MAPE)** | < 5% on test data | Unit |
| **Test Coverage** | > 90% | All |

## 🛠️ Test Configuration

### Environment Variables

```bash
export TESTING=1                    # Enable test mode
export CUDA_VISIBLE_DEVICES=0       # GPU device selection
export PYTHONPATH=.                 # Python path configuration
```

### Custom Markers

Use pytest markers to run specific test types:

```bash
# Test markers available
pytest tests/neural/ -m "unit"          # Unit tests only
pytest tests/neural/ -m "integration"   # Integration tests only
pytest tests/neural/ -m "performance"   # Performance tests only
pytest tests/neural/ -m "gpu"           # GPU tests only
pytest tests/neural/ -m "slow"          # Long-running tests
pytest tests/neural/ -m "stress"        # Stress tests
pytest tests/neural/ -m "regression"    # Regression tests
```

### Configuration Files

- **`pytest.ini`** - PyTest configuration and markers
- **`conftest.py`** - Shared fixtures and test setup
- **`utils/fixtures.py`** - Reusable test fixtures

## 🔧 Mock Components

When neural components are not available, the test suite uses comprehensive mocks:

- **MockNHITSModel** - Simulates NHITS neural network
- **MockRealTimeEngine** - Simulates inference engine
- **MockMCPServer** - Simulates MCP server with neural tools
- **MockMultiAssetProcessor** - Simulates parallel processing

This ensures tests can run in any environment while validating integration patterns.

## 📈 Coverage Reporting

Generate comprehensive coverage reports:

```bash
# HTML coverage report
pytest tests/neural/ --cov=src --cov-report=html:htmlcov/neural

# Terminal coverage report
pytest tests/neural/ --cov=src --cov-report=term-missing

# XML coverage report (CI/CD)
pytest tests/neural/ --cov=src --cov-report=xml:coverage-neural.xml

# Coverage with minimum threshold
pytest tests/neural/ --cov=src --cov-fail-under=90
```

View the HTML report at: `htmlcov/neural/index.html`

## 🎯 Test Data Generation

The test suite includes sophisticated data generators:

### Synthetic Time Series
```python
from tests.neural.utils.data_generators import SyntheticTimeSeriesGenerator

generator = SyntheticTimeSeriesGenerator(seed=42)
params = TimeSeriesParams(
    n_points=1000,
    trend=0.01,
    seasonality_periods=[24, 168],  # Daily and weekly patterns
    noise_level=0.1
)
series = generator.generate_single_series(params)
```

### Market Scenarios
```python
from tests.neural.utils.data_generators import MarketScenarioGenerator

generator = MarketScenarioGenerator()
bull_market = generator.generate_scenario('bull_market', n_points=1000)
bear_market = generator.generate_scenario('bear_market', n_points=1000)
crash_scenario = generator.generate_scenario('market_crash', n_points=500)
```

### News Events
```python
from tests.neural.utils.data_generators import NewsEventGenerator

generator = NewsEventGenerator(seed=42)
events = generator.generate_events(
    start_time=datetime.now() - timedelta(days=7),
    end_time=datetime.now(),
    assets=['AAPL', 'GOOGL', 'MSFT']
)
```

## 🔍 Debugging Tests

### Verbose Output
```bash
# Detailed test output
pytest tests/neural/ -v -s

# Show test durations
pytest tests/neural/ --durations=10

# Stop on first failure
pytest tests/neural/ -x

# Show local variables on failure
pytest tests/neural/ -l
```

### GPU Debugging
```bash
# Check GPU availability
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"

# Monitor GPU usage during tests
nvidia-smi -l 1

# Run GPU tests with memory monitoring
pytest tests/neural/ -m gpu -s
```

### Performance Debugging
```bash
# Profile test performance
pytest tests/neural/ --profile

# Memory profiling
pytest tests/neural/ -m performance --memray

# Benchmark specific tests
pytest tests/neural/performance/test_inference_performance.py::TestInferenceLatency::test_single_prediction_latency -v
```

## 🚨 Troubleshooting

### Common Issues

**GPU Tests Failing**
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Update GPU drivers
sudo apt update && sudo apt install nvidia-driver-535
```

**Memory Issues**
```bash
# Increase swap space
sudo swapon --show
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**Import Errors**
```bash
# Check Python path
export PYTHONPATH=/workspaces/ai-news-trader:$PYTHONPATH

# Install in development mode
pip install -e .
```

**Performance Tests Timing Out**
```bash
# Increase timeout
pytest tests/neural/ --timeout=600

# Run quick mode
python tests/neural/run_tests.py --all --quick
```

## 📚 Test Examples

### Basic Unit Test
```python
def test_nhits_model_creation(basic_nhits_config):
    \"\"\"Test NHITS model creation.\"\"\"
    model = OptimizedNHITS(basic_nhits_config)
    assert isinstance(model, nn.Module)
    assert model.config == basic_nhits_config
```

### Performance Test
```python
@benchmark_latency()
def test_inference_latency(basic_nhits_config, device):
    \"\"\"Test inference latency performance.\"\"\"
    model = OptimizedNHITS(basic_nhits_config).to(device)
    input_tensor = torch.randn(1, basic_nhits_config.input_size, device=device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    assert output['point_forecast'].shape[1] == basic_nhits_config.h
```

### GPU Test
```python
@skip_if_no_gpu
def test_gpu_acceleration(basic_nhits_config):
    \"\"\"Test GPU acceleration.\"\"\"
    model = OptimizedNHITS(basic_nhits_config).cuda()
    input_tensor = torch.randn(16, basic_nhits_config.input_size, device='cuda')
    
    with torch.no_grad():
        output = model(input_tensor)
    
    assert output['point_forecast'].device.type == 'cuda'
```

## 🔄 Continuous Integration

### GitHub Actions Example
```yaml
name: Neural Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements-neural.txt
      - run: python tests/neural/run_tests.py --all --quick
      - run: python tests/neural/run_tests.py --coverage
      - uses: codecov/codecov-action@v3
        with:
          file: coverage-neural.xml
```

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: neural-tests
        name: Neural Tests
        entry: python tests/neural/run_tests.py --unit --integration
        language: system
        pass_filenames: false
```

## 📋 Test Checklist

Before merging neural forecasting code, ensure:

- [ ] All unit tests pass
- [ ] Integration tests pass  
- [ ] Performance tests meet targets
- [ ] GPU tests pass (if GPU available)
- [ ] Test coverage > 90%
- [ ] No memory leaks detected
- [ ] Documentation updated
- [ ] Performance baselines updated (if applicable)

## 🤝 Contributing

### Adding New Tests

1. **Choose appropriate category** (unit/integration/performance)
2. **Use existing fixtures** from `utils/fixtures.py`
3. **Follow naming conventions** (`test_*.py`, `Test*`, `test_*`)
4. **Add appropriate markers** (`@pytest.mark.gpu`, etc.)
5. **Include docstrings** explaining test purpose
6. **Update this README** if adding new test categories

### Test Guidelines

- **Test one thing at a time** - Focus on single functionality
- **Use descriptive names** - Clear test purpose from name
- **Mock external dependencies** - Use mocks for external services
- **Clean up resources** - Ensure proper cleanup after tests
- **Performance considerations** - Set appropriate timeouts
- **Documentation** - Comment complex test logic

## 📞 Support

For questions about the neural test suite:

1. **Check this README** for common patterns
2. **Review existing tests** for examples
3. **Check test logs** for detailed error information
4. **Run individual test files** to isolate issues
5. **Use verbose mode** (`-v -s`) for debugging

## 📄 License

This test suite is part of the AI News Trading Platform and follows the same license terms.

---

**Happy Testing! 🚀**

The neural forecasting test suite ensures robust, performant, and reliable neural forecasting capabilities for the AI News Trading Platform.