# Conformalized Quantile Regression (CQR) Module

## Overview

Production-grade implementation of Conformalized Quantile Regression for constructing distribution-free prediction intervals with finite-sample coverage guarantees.

## Quick Start

```rust
use ats_core::cqr::{CqrConfig, CqrCalibrator};

// Configure for 90% coverage
let config = CqrConfig {
    alpha: 0.1,
    symmetric: true,
};

let mut calibrator = CqrCalibrator::new(config);

// Calibration data from quantile regression model
let y_cal = vec![5.0, 5.2, 4.8, 5.1, 4.9];
let q_lo_cal = vec![4.5, 4.7, 4.3, 4.6, 4.4];
let q_hi_cal = vec![5.5, 5.7, 5.3, 5.6, 5.4];

// Calibrate
calibrator.calibrate(&y_cal, &q_lo_cal, &q_hi_cal);

// Predict interval for new sample
let (lower, upper) = calibrator.predict_interval(4.5, 5.5);
println!("90% prediction interval: [{:.2}, {:.2}]", lower, upper);

// Validate coverage
let coverage = calibrator.compute_coverage(&y_test, &q_lo_test, &q_hi_test);
assert!(coverage >= 0.9);
```

## Features

### Core Capabilities

- **Distribution-Free Coverage** - No parametric assumptions
- **Finite-Sample Guarantees** - Valid for any sample size
- **Exchangeability Only** - Minimal statistical assumptions
- **Symmetric & Asymmetric Variants** - Flexible interval construction

### Variants

#### 1. Symmetric CQR (base.rs)

Standard CQR with equal corrections on both sides.

```rust
let mut calibrator = CqrCalibrator::new(CqrConfig {
    alpha: 0.1,
    symmetric: true,
});
```

#### 2. Asymmetric CQR (asymmetric.rs)

Separate lower and upper corrections for potentially tighter intervals.

```rust
use ats_core::cqr::{AsymmetricCqrCalibrator, AsymmetricCqrConfig};

let mut calibrator = AsymmetricCqrCalibrator::new(AsymmetricCqrConfig {
    alpha: 0.1,
    alpha_lo: 0.05,
    alpha_hi: 0.05,
});
```

#### 3. Enhanced Symmetric CQR (symmetric.rs)

With diagnostic utilities:

```rust
use ats_core::cqr::SymmetricCqr;

let mut cqr = SymmetricCqr::new(config);
cqr.calibrate(&y_cal, &q_lo_cal, &q_hi_cal);

// Get interval statistics
let stats = cqr.compute_interval_statistics(&q_lo, &q_hi);
println!("Mean width: {:.2}", stats.mean_width);

// Evaluate performance
let metrics = cqr.evaluate(&y_test, &q_lo_test, &q_hi_test);
println!("Coverage: {:.1}%", metrics.coverage * 100.0);
println!("Efficiency: {:.3}", metrics.efficiency);
```

## Mathematical Foundation

### Algorithm (Romano et al., 2019)

**Nonconformity Score:**
```
E(x, y) = max(q̂_lo(x) - y, y - q̂_hi(x))
```

**Calibration:**
1. Compute scores E₁, ..., Eₙ on calibration set
2. Determine threshold: Q̂ = Quantile((1-α)(1 + 1/n), {E₁, ..., Eₙ})
3. Prediction interval: C(x) = [q̂_lo(x) - Q̂, q̂_hi(x) + Q̂]

**Coverage Guarantee:**
```
P(Y ∈ C(X)) ≥ 1 - α
```

under exchangeability assumption.

## API Reference

### CqrCalibrator

```rust
pub struct CqrCalibrator {
    // ...
}

impl CqrCalibrator {
    /// Create new calibrator
    pub fn new(config: CqrConfig) -> Self

    /// Compute nonconformity score
    pub fn nonconformity_score(&self, y: f32, q_lo: f32, q_hi: f32) -> f32

    /// Calibrate on held-out set
    pub fn calibrate(&mut self, y_cal: &[f32], q_lo_cal: &[f32], q_hi_cal: &[f32])

    /// Predict interval for single sample
    pub fn predict_interval(&self, q_lo: f32, q_hi: f32) -> (f32, f32)

    /// Batch prediction intervals
    pub fn predict_intervals_batch(
        &self,
        q_lo_batch: &[f32],
        q_hi_batch: &[f32]
    ) -> Vec<(f32, f32)>

    /// Compute empirical coverage
    pub fn compute_coverage(
        &self,
        y_test: &[f32],
        q_lo_test: &[f32],
        q_hi_test: &[f32]
    ) -> f32
}
```

### Configuration

```rust
pub struct CqrConfig {
    /// Miscoverage level α ∈ (0, 1)
    pub alpha: f32,
    /// Whether to use symmetric intervals
    pub symmetric: bool,
}
```

## Use Cases

### 1. Financial Risk Assessment

```rust
// Predict stock price intervals with 95% confidence
let config = CqrConfig { alpha: 0.05, symmetric: true };
let mut calibrator = CqrCalibrator::new(config);

// Train quantile regression on historical data
// ... (get q_lo and q_hi predictions)

calibrator.calibrate(&historical_prices, &q_lo_cal, &q_hi_cal);

// Forecast tomorrow's price range
let (lower_bound, upper_bound) = calibrator.predict_interval(q_lo_tomorrow, q_hi_tomorrow);
```

### 2. Medical Predictions

```rust
// Predict patient recovery time with uncertainty
let config = CqrConfig { alpha: 0.1, symmetric: true };
let mut calibrator = CqrCalibrator::new(config);

calibrator.calibrate(&patient_outcomes, &q_lo_pred, &q_hi_pred);

let (min_recovery, max_recovery) = calibrator.predict_interval(q_lo, q_hi);
println!("90% chance recovery in {:.0}-{:.0} days", min_recovery, max_recovery);
```

### 3. Demand Forecasting

```rust
// Forecast product demand with safety margins
let config = AsymmetricCqrConfig {
    alpha: 0.1,
    alpha_lo: 0.05,  // Conservative on low end
    alpha_hi: 0.05,
};
let mut calibrator = AsymmetricCqrCalibrator::new(config);

calibrator.calibrate(&historical_demand, &q_lo_cal, &q_hi_cal);

let (low_estimate, high_estimate) = calibrator.predict_interval(q_lo, q_hi);
```

## Performance

### Computational Complexity

- **Calibration:** O(n log n) - dominated by sorting
- **Prediction:** O(1) per interval
- **Batch Prediction:** O(m) for m samples

### Benchmarks (10k calibration, 1k test)

```
Calibration:    <1000ms
Batch predict:  <10ms
Per prediction: <10μs
```

## Testing

### Run Tests

```bash
cargo test -p ats-core --lib cqr
```

### Integration Tests

```bash
cargo test -p ats-core --test cqr_integration_test
```

### Coverage Report

All tests include:
- ✅ Coverage validation
- ✅ Mathematical correctness
- ✅ Edge case handling
- ✅ Performance benchmarks

## References

### Academic Papers

1. **Romano, Y., Patterson, E., & Candès, E. (2019)**
   - "Conformalized Quantile Regression"
   - Advances in Neural Information Processing Systems 32
   - [arXiv:1905.03222](https://arxiv.org/abs/1905.03222)

2. **Sesia, M. & Candès, E.J. (2020)**
   - "A comparison of some conformal quantile regression methods"
   - Stat, 9(1), e261
   - [DOI:10.1002/sta4.261](https://doi.org/10.1002/sta4.261)

3. **Feldman, S., Bates, S., & Romano, Y. (2021)**
   - "Improving Conditional Coverage via Orthogonal Quantile Regression"
   - [arXiv:2106.00088](https://arxiv.org/abs/2106.00088)

### Additional Resources

- [Conformal Prediction Tutorial](https://people.eecs.berkeley.edu/~angelopoulos/blog/posts/conformal-prediction/)
- [CQR Python Implementation](https://github.com/yromano/cqr)

## Module Structure

```
src/cqr/
├── mod.rs              # Module exports and integration tests
├── base.rs             # Core symmetric CQR (CqrCalibrator)
├── asymmetric.rs       # Asymmetric variant (AsymmetricCqrCalibrator)
├── symmetric.rs        # Enhanced symmetric with diagnostics (SymmetricCqr)
├── calibration.rs      # Quantile utilities
└── README.md           # This file
```

## Examples

See `tests/cqr_integration_test.rs` for comprehensive examples including:
- Coverage guarantee validation
- Symmetric vs asymmetric comparison
- Diagnostic utilities
- Performance benchmarking
- Edge case handling

## License

Part of the ATS-Core library. See repository LICENSE for details.

## Contributing

When contributing:
1. Maintain mathematical rigor
2. Include academic citations
3. Add comprehensive tests
4. Update documentation
5. Follow existing code style

## Support

For issues or questions:
- GitHub Issues: [HyperPhysics Issues](https://github.com/your-repo/issues)
- Documentation: See `/docs` directory

---

**Status:** ✅ Production Ready
**Version:** 1.0.0
**Last Updated:** 2025-11-27
