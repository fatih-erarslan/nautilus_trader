# Neural/ML Test Suite

Comprehensive test coverage for neural network functions in neural-trader.

## Quick Start

```bash
# Run all ML tests
npm test tests/ml/

# Run validation tests only
npm test tests/ml/neural-validation.test.js

# Run performance tests only
npm test tests/ml/model-performance.test.js

# Run with coverage
npm test tests/ml/ -- --coverage
```

## Test Files

### 1. Neural Validation Tests (`neural-validation.test.js`)

**Purpose**: Validate all neural functions with various parameters and edge cases

**Coverage**:
- ✅ neuralForecast (8 tests)
- ✅ neuralTrain (10 tests)
- ✅ neuralEvaluate (7 tests)
- ✅ neuralModelStatus (5 tests)
- ✅ neuralOptimize (6 tests)
- ✅ neuralBacktest (8 tests)
- ✅ Integration workflow (1 test)

**Total**: 45 test cases

**Run time**: ~5 minutes

### 2. Model Performance Tests (`model-performance.test.js`)

**Purpose**: Benchmark training speed, inference latency, and resource usage

**Coverage**:
- ✅ GPU vs CPU benchmarks
- ✅ Data size scaling (1K-1M samples)
- ✅ Inference latency (P50, P95, P99)
- ✅ Memory profiling
- ✅ Batch processing efficiency
- ✅ System resource monitoring

**Total**: 20 test cases

**Run time**: ~10 minutes (includes benchmarks)

## Test Data

Tests automatically generate synthetic data in `/tests/fixtures/`:
- `training_data.csv` - 1000 samples for training
- `test_data.csv` - 200 samples for testing
- `benchmark_*.csv` - Various sizes for performance tests

## Expected Results

### Validation Tests

All tests should **PASS** with:
- Forecast accuracy: R² > 0.70
- Training convergence: Loss decreasing
- Metrics relationships: RMSE ≥ MAE
- No errors or crashes

### Performance Tests

Typical results on modern hardware:

**Training Speed (5000 samples, 100 epochs)**:
- GRU: 30-40s (GPU) / 4-6min (CPU)
- LSTM: 40-60s (GPU) / 7-9min (CPU)
- Transformer: 80-120s (GPU) / 14-18min (CPU)

**Inference Latency (100 predictions)**:
- Average: 50-100ms (GPU) / 300-500ms (CPU)
- P95: 80-120ms (GPU) / 450-650ms (CPU)

**Memory Usage**:
- GRU: ~150MB training / ~50MB inference
- LSTM: ~250MB training / ~75MB inference
- Transformer: ~500MB training / ~150MB inference

## Troubleshooting

### Tests Failing

**Issue**: "Model not trained"
**Solution**: Ensure training completes successfully before evaluation

**Issue**: "Data not found"
**Solution**: Tests auto-generate data; check write permissions

**Issue**: Timeout errors
**Solution**: Increase Jest timeout: `jest.setTimeout(120000)`

### Performance Issues

**Issue**: Slow training
**Solution**: Enable GPU with `use_gpu: true`

**Issue**: Memory errors
**Solution**: Reduce batch size or model complexity

**Issue**: Latency too high
**Solution**: Use GPU for inference, batch predictions

## Test Configuration

```javascript
// jest.config.js
module.exports = {
  testTimeout: 120000,  // 2 minutes per test
  testEnvironment: 'node',
  coveragePathIgnorePatterns: ['/node_modules/', '/tests/fixtures/']
};
```

## Continuous Integration

Tests are designed for CI/CD pipelines:

```yaml
# .github/workflows/test.yml
- name: Run ML tests
  run: npm test tests/ml/
  timeout-minutes: 20
```

## Contributing

When adding new neural functions:

1. Add validation tests in `neural-validation.test.js`
2. Add performance benchmarks in `model-performance.test.js`
3. Update documentation
4. Ensure 100% test coverage

## Related Documentation

- [Neural Network Guide](../../docs/ml/neural-network-guide.md)
- [Training Best Practices](../../docs/ml/training-best-practices.md)
- [Complete Pipeline Example](../../examples/ml/complete-training-pipeline.js)
