# Neuro-Divergent Testing Plan

## Overview

Comprehensive test suite for validating the Rust port of NeuralForecast against Python baseline.

## Test Categories

### 1. Accuracy Validation Tests (`tests/accuracy_validation.rs`)

**Objective**: Ensure Rust implementations match Python NeuralForecast with ε < 1e-4

**Tests**:
- ✅ `test_nhits_accuracy_vs_python` - NHITS model accuracy
- ✅ `test_nbeats_accuracy_vs_python` - NBEATS model accuracy
- ✅ `test_tft_accuracy_vs_python` - TFT model accuracy
- ✅ `test_lstm_accuracy_vs_python` - LSTM model accuracy
- ✅ `test_gru_accuracy_vs_python` - GRU model accuracy
- ✅ `test_prediction_intervals_accuracy` - 80%, 90%, 95% intervals
- ✅ `test_all_models_accuracy_batch` - Batch test all 27+ models
- ✅ `test_numerical_stability` - Stability across input ranges
- ✅ `test_model_persistence` - Save/load preserves accuracy

**Python Baseline Generation**:
```python
# generate_baselines.py
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, NBEATS, TFT, LSTM, GRU
import json

# Generate reference predictions
model = NHITS(...)
forecast = model.fit(train_data)
predictions = forecast.predict(horizon=24)

# Save to JSON
baseline = {
    "model_name": "nhits",
    "predictions": predictions.tolist(),
    "intervals_80": (lower_80.tolist(), upper_80.tolist()),
    "intervals_90": (lower_90.tolist(), upper_90.tolist()),
    "intervals_95": (lower_95.tolist(), upper_95.tolist()),
    "training_time_ms": training_time,
    "inference_time_ms": inference_time
}

with open("test-data/python-baselines/nhits.json", "w") as f:
    json.dump(baseline, f)
```

### 2. Performance Benchmarks (`tests/performance_benchmarks.rs`)

**Objective**: Validate 3-5x speedup vs Python

**Benchmarks**:
- ✅ `benchmark_nhits_training_speed` - Training performance (target: 3-5x)
- ✅ `benchmark_nhits_inference_speed` - Inference latency (target: 2-4x)
- ✅ `benchmark_batch_inference` - Batch throughput
- ✅ `benchmark_memory_usage` - Memory footprint (target: <250MB)
- ✅ `benchmark_cross_validation_performance` - CV speed
- ✅ `benchmark_all_models_comparison` - Model comparison table
- ✅ `stress_test_large_dataset` - 10k, 50k, 100k, 500k samples

**Expected Results**:
| Model | Python Baseline | Rust Target | Speedup |
|-------|----------------|-------------|---------|
| NHITS Training | 45s | 11.8s | 3.8x |
| NHITS Inference | 120ms | 38ms | 3.2x |
| LSTM Training | 60s | 15s | 4.0x |
| TFT Training | 90s | 22s | 4.1x |

### 3. Integration Tests (`tests/integration_tests.rs`)

**Objective**: Validate end-to-end workflows

**Tests**:
- ✅ `test_end_to_end_forecast_workflow` - Full pipeline
- ✅ `test_multi_model_ensemble` - Ensemble predictions
- ✅ `test_cross_validation_pipeline` - 5-fold CV
- ✅ `test_model_save_and_load` - Persistence
- ✅ `test_incremental_learning` - Model updates
- ✅ `test_batch_prediction` - Multiple horizons
- ✅ `test_prediction_with_exogenous_variables` - External features

### 4. Property-Based Tests (`tests/property_tests.rs`)

**Objective**: Verify model invariants hold

**Properties**:
- ✅ `prop_predictions_are_finite` - No NaN/Inf
- ✅ `prop_prediction_length_matches_horizon` - Correct output size
- ✅ `prop_intervals_properly_ordered` - lower <= upper
- ✅ `prop_higher_confidence_wider_intervals` - 95% > 80%
- ✅ `prop_deterministic_predictions` - Same input = same output
- ✅ `prop_more_data_not_worse` - More data helps
- ✅ `qc_rejects_zero_horizon` - Input validation
- ✅ `qc_rejects_empty_data` - Input validation
- ✅ `prop_metadata_accuracy` - Metadata correctness
- ✅ `prop_save_load_preserves_predictions` - Persistence correctness
- ✅ `prop_scaling_invariance` - Scale handling

### 5. Market Data Validation (`tests/market_data_validation.rs`)

**Objective**: Real-world performance on financial data

**Tests**:
- ✅ `test_sp500_forecasting_accuracy` - S&P 500 (MAPE < 20%)
- ✅ `test_bitcoin_volatility_forecasting` - BTC-USD high volatility
- ✅ `test_forex_intraday_prediction` - EUR/USD intraday
- ✅ `test_commodity_seasonality` - Gold seasonality
- ✅ `test_multi_asset_ensemble` - Cross-asset predictions
- ✅ `test_crisis_period_robustness` - 2008 crisis data

**Metrics**:
- MAPE (Mean Absolute Percentage Error)
- RMSE (Root Mean Squared Error)
- Directional Accuracy
- Interval Coverage (95% interval should capture ~95% of actuals)

### 6. Criterion Benchmarks (`benches/model_benchmarks.rs`)

**Objective**: Detailed performance profiling

**Benchmark Groups**:
- `training` - Training speed across sample sizes
- `inference` - Inference latency across horizons
- `batch_inference` - Batch prediction throughput
- `preprocessing` - Data normalization speed
- `cross_validation` - CV performance

## Test Execution

### Run All Tests
```bash
# Run all unit tests
cargo test -p neuro-divergent

# Run ignored tests (requires Python baselines)
cargo test -p neuro-divergent -- --ignored --nocapture

# Run benchmarks
cargo bench -p neuro-divergent

# Run with coverage
cargo tarpaulin --out Html --output-dir coverage -p neuro-divergent
```

### Run Specific Test Categories
```bash
# Accuracy validation only
cargo test -p neuro-divergent accuracy_validation

# Performance benchmarks only
cargo test -p neuro-divergent performance_benchmarks -- --ignored --nocapture

# Integration tests only
cargo test -p neuro-divergent integration_tests

# Property-based tests
cargo test -p neuro-divergent property_tests

# Market data validation
cargo test -p neuro-divergent market_data_validation -- --ignored
```

## Coverage Targets

- **Overall Coverage**: ≥95%
- **Statement Coverage**: ≥95%
- **Branch Coverage**: ≥90%
- **Function Coverage**: ≥95%

## Performance Targets

### Speed
- Training: 3-5x faster than Python
- Inference: 2-4x faster than Python
- Preprocessing: 5-10x faster than Python

### Memory
- Training (10k samples): <250MB
- Inference (batch 32): <50MB
- Model size: <100MB

### Accuracy
- Max error vs Python: ε < 1e-4
- MAPE on S&P 500: <20%
- Interval coverage: 90-98% for 95% intervals

## Test Data Requirements

### Python Baselines
```
test-data/
├── python-baselines/
│   ├── nhits.json
│   ├── nbeats.json
│   ├── tft.json
│   ├── lstm.json
│   ├── gru.json
│   └── ... (27+ models)
```

### Market Data
```
test-data/
├── market/
│   ├── SPY.csv        # S&P 500
│   ├── BTC-USD.csv    # Bitcoin
│   ├── EURUSD.csv     # EUR/USD Forex
│   ├── GLD.csv        # Gold
│   └── SPY_2008.csv   # Crisis period data
```

## Continuous Integration

### GitHub Actions Workflow
```yaml
name: Neuro-Divergent Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Run tests
        run: cargo test -p neuro-divergent
      - name: Run benchmarks
        run: cargo bench -p neuro-divergent --no-run
      - name: Coverage
        run: |
          cargo install cargo-tarpaulin
          cargo tarpaulin --out Xml -p neuro-divergent
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Test Results Report Template

```markdown
# Neuro-Divergent Test Results

## Summary
- Total Tests: XXX
- Passed: XXX
- Failed: XXX
- Ignored: XXX
- Coverage: XX.X%

## Accuracy Validation
| Model | Max Error | Status |
|-------|-----------|--------|
| NHITS | 8.3e-5 | ✅ PASS |
| NBEATS | 7.1e-5 | ✅ PASS |
| TFT | 9.2e-5 | ✅ PASS |
| ... | ... | ... |

## Performance Benchmarks
| Benchmark | Python | Rust | Speedup |
|-----------|--------|------|---------|
| NHITS Training | 45s | 11.8s | 3.8x |
| NHITS Inference | 120ms | 38ms | 3.2x |
| ... | ... | ... | ... |

## Market Data Results
| Asset | MAPE | RMSE | Dir. Accuracy |
|-------|------|------|---------------|
| S&P 500 | 12.3% | 45.2 | 68.4% |
| Bitcoin | 18.7% | 1234.5 | 62.1% |
| ... | ... | ... | ... |

## Issues Found
- [ ] Issue 1: Description
- [ ] Issue 2: Description

## Recommendations
1. Optimization opportunity in...
2. Consider additional tests for...
```

## Next Steps

1. **Generate Python baselines** - Run Python NeuralForecast to create reference data
2. **Implement placeholder models** - Basic NHITS, NBEATS, TFT, LSTM implementations
3. **Run initial tests** - Execute test suite to identify failures
4. **Iterate on accuracy** - Tune implementations to match Python baseline
5. **Optimize performance** - Profile and optimize to meet speed targets
6. **Document results** - Generate comprehensive test report
