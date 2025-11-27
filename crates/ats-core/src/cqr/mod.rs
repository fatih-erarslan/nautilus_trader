//! Conformalized Quantile Regression (CQR) Module
//!
//! This module implements Conformalized Quantile Regression for constructing
//! distribution-free prediction intervals with finite-sample coverage guarantees.
//!
//! # Overview
//!
//! CQR (Romano et al., 2019) provides a framework for producing prediction intervals
//! that achieve valid coverage under minimal assumptions:
//!
//! - **Distribution-free**: No parametric assumptions about data distribution
//! - **Finite-sample guarantees**: Coverage holds for any sample size
//! - **Exchangeability only**: Only requires i.i.d. or exchangeable data
//!
//! # Mathematical Framework
//!
//! Given a base quantile regression model producing quantiles at levels α/2 and 1-α/2,
//! CQR calibrates these quantiles using a held-out calibration set to achieve exact
//! coverage 1-α.
//!
//! ## Basic Algorithm
//!
//! 1. Split data into training, calibration, and test sets
//! 2. Train quantile regression on training set
//! 3. Compute nonconformity scores on calibration set
//! 4. Determine correction factor via quantile of scores
//! 5. Apply correction to produce final prediction intervals
//!
//! # Modules
//!
//! - [`base`]: Core symmetric CQR implementation
//! - [`asymmetric`]: Asymmetric CQR with separate lower/upper corrections
//! - [`symmetric`]: Enhanced symmetric CQR with diagnostics
//! - [`calibration`]: Utilities for quantile calibration and validation
//!
//! # Example
//!
//! ```rust
//! use ats_core::cqr::{CqrConfig, CqrCalibrator};
//!
//! // Configure CQR for 90% coverage
//! let config = CqrConfig {
//!     alpha: 0.1,
//!     symmetric: true,
//! };
//!
//! let mut calibrator = CqrCalibrator::new(config);
//!
//! // Calibration data (from held-out set)
//! let y_cal = vec![5.0, 5.2, 4.8, 5.1, 4.9];
//! let q_lo_cal = vec![4.5, 4.7, 4.3, 4.6, 4.4];
//! let q_hi_cal = vec![5.5, 5.7, 5.3, 5.6, 5.4];
//!
//! calibrator.calibrate(&y_cal, &q_lo_cal, &q_hi_cal);
//!
//! // Make prediction
//! let (lower, upper) = calibrator.predict_interval(4.5, 5.5);
//! println!("90% prediction interval: [{}, {}]", lower, upper);
//! ```
//!
//! # References
//!
//! - Romano, Y., Patterson, E., & Candès, E. (2019). "Conformalized Quantile Regression"
//!   Advances in Neural Information Processing Systems 32.
//! - Sesia, M. & Candès, E.J. (2020). "A comparison of some conformal quantile regression methods"
//!   Stat, 9(1), e261.
//! - Feldman, S., Bates, S., & Romano, Y. (2021). "Improving Conditional Coverage via
//!   Orthogonal Quantile Regression"

pub mod asymmetric;
pub mod base;
pub mod calibration;
pub mod symmetric;

// Re-export main types
pub use asymmetric::{AsymmetricCqrCalibrator, AsymmetricCqrConfig};
pub use base::{CqrCalibrator, CqrConfig};
pub use calibration::{
    compute_quantile, compute_quantiles, interval_width_stats, stratified_coverage,
    validate_coverage,
};
pub use symmetric::{EvaluationMetrics, IntervalStatistics, SymmetricCqr};

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// End-to-end test of CQR workflow
    #[test]
    fn test_cqr_full_workflow() {
        // Generate synthetic data
        let n = 100;
        let mut y_train = Vec::new();
        let mut y_cal = Vec::new();
        let mut y_test = Vec::new();

        // Training set (60%)
        for i in 0..60 {
            y_train.push((i as f32) / 10.0);
        }

        // Calibration set (20%)
        for i in 60..80 {
            y_cal.push((i as f32) / 10.0);
        }

        // Test set (20%)
        for i in 80..100 {
            y_test.push((i as f32) / 10.0);
        }

        // Simulate quantile predictions (in practice, from trained model)
        let q_lo_cal: Vec<f32> = y_cal.iter().map(|y| y - 0.5).collect();
        let q_hi_cal: Vec<f32> = y_cal.iter().map(|y| y + 0.5).collect();

        let q_lo_test: Vec<f32> = y_test.iter().map(|y| y - 0.5).collect();
        let q_hi_test: Vec<f32> = y_test.iter().map(|y| y + 0.5).collect();

        // Setup CQR
        let config = CqrConfig {
            alpha: 0.1,
            symmetric: true,
        };
        let mut calibrator = CqrCalibrator::new(config);

        // Calibrate
        calibrator.calibrate(&y_cal, &q_lo_cal, &q_hi_cal);

        // Evaluate coverage
        let coverage = calibrator.compute_coverage(&y_test, &q_lo_test, &q_hi_test);

        // Coverage should meet target
        assert!(
            coverage >= 0.9,
            "Coverage {} below target 0.9",
            coverage
        );
    }

    /// Compare symmetric and asymmetric CQR
    #[test]
    fn test_symmetric_vs_asymmetric() {
        let y_cal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let q_lo_cal = vec![0.5, 1.5, 2.5, 3.5, 4.5];
        let q_hi_cal = vec![1.5, 2.5, 3.5, 4.5, 5.5];

        // Symmetric CQR
        let config_sym = CqrConfig {
            alpha: 0.1,
            symmetric: true,
        };
        let mut sym = CqrCalibrator::new(config_sym);
        sym.calibrate(&y_cal, &q_lo_cal, &q_hi_cal);

        // Asymmetric CQR
        let config_asym = AsymmetricCqrConfig {
            alpha: 0.1,
            alpha_lo: 0.05,
            alpha_hi: 0.05,
        };
        let mut asym = AsymmetricCqrCalibrator::new(config_asym);
        asym.calibrate(&y_cal, &q_lo_cal, &q_hi_cal);

        // Compare intervals
        let (sym_lo, sym_hi) = sym.predict_interval(2.0, 3.0);
        let (asym_lo, asym_hi) = asym.predict_interval(2.0, 3.0);

        // Both should produce valid intervals
        assert!(sym_lo < sym_hi);
        assert!(asym_lo < asym_hi);

        // Intervals should be roughly similar
        let sym_width = sym_hi - sym_lo;
        let asym_width = asym_hi - asym_lo;

        // Allow 20% difference
        assert!(
            (sym_width / asym_width - 1.0).abs() < 0.2,
            "Interval widths differ significantly"
        );
    }
}
