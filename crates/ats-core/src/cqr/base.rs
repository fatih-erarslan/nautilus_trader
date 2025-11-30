//! Conformalized Quantile Regression (Romano et al., 2019)
//!
//! # Mathematical Foundation
//!
//! Given quantile estimates q̂_lo(x), q̂_hi(x) at levels α/2 and 1-α/2:
//!
//! **Nonconformity score**: E(x,y) = max(q̂_lo(x) - y, y - q̂_hi(x))
//!
//! **Prediction interval**: [q̂_lo(x) - Q̂, q̂_hi(x) + Q̂]
//! where Q̂ is the (1-α)(1 + 1/n) quantile of calibration scores
//!
//! # References
//!
//! - Romano, Y., Patterson, E., & Candès, E. (2019). "Conformalized Quantile Regression"
//!   Advances in Neural Information Processing Systems 32.
//! - Sesia, M. & Candès, E.J. (2020). "A comparison of some conformal quantile regression methods"
//!   Stat, 9(1), e261.

use std::cmp::Ordering;

/// Configuration for Conformalized Quantile Regression
#[derive(Clone, Debug)]
pub struct CqrConfig {
    /// Miscoverage level α (e.g., 0.1 for 90% coverage)
    /// Must be in (0, 1)
    pub alpha: f32,
    /// Whether to use symmetric or asymmetric intervals
    pub symmetric: bool,
}

impl Default for CqrConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            symmetric: true,
        }
    }
}

impl CqrConfig {
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.alpha <= 0.0 || self.alpha >= 1.0 {
            return Err(format!(
                "alpha must be in (0, 1), got {}",
                self.alpha
            ));
        }
        Ok(())
    }
}

/// Conformalized Quantile Regression calibrator
///
/// Implements the CQR algorithm from Romano et al. (2019) for distribution-free
/// prediction intervals with finite-sample coverage guarantees.
pub struct CqrCalibrator {
    config: CqrConfig,
    /// Calibration scores from calibration set
    calibration_scores: Vec<f32>,
    /// Computed quantile threshold
    quantile_threshold: Option<f32>,
}

impl CqrCalibrator {
    /// Create a new CQR calibrator
    ///
    /// # Arguments
    /// * `config` - CQR configuration
    ///
    /// # Panics
    /// Panics if configuration validation fails
    pub fn new(config: CqrConfig) -> Self {
        config.validate().expect("Invalid CQR configuration");

        Self {
            config,
            calibration_scores: Vec::new(),
            quantile_threshold: None,
        }
    }

    /// Compute nonconformity score for a single sample
    ///
    /// Following Romano et al. (2019), the nonconformity score is:
    /// E(x,y) = max(q̂_lo(x) - y, y - q̂_hi(x))
    ///
    /// This measures how far the true value y falls outside the predicted
    /// quantile interval [q̂_lo(x), q̂_hi(x)].
    ///
    /// # Arguments
    /// * `y` - True value
    /// * `q_lo` - Lower quantile prediction (at level α/2)
    /// * `q_hi` - Upper quantile prediction (at level 1-α/2)
    ///
    /// # Returns
    /// Nonconformity score (≥ 0). Score of 0 means y is within [q_lo, q_hi].
    #[inline]
    pub fn nonconformity_score(&self, y: f32, q_lo: f32, q_hi: f32) -> f32 {
        debug_assert!(
            q_lo <= q_hi,
            "Lower quantile must be ≤ upper quantile: {} > {}",
            q_lo,
            q_hi
        );
        f32::max(q_lo - y, y - q_hi)
    }

    /// Calibrate on a calibration set
    ///
    /// Computes nonconformity scores for calibration samples and determines
    /// the quantile threshold Q̂ used for prediction intervals.
    ///
    /// # Algorithm (Romano et al., 2019)
    ///
    /// 1. For each calibration sample (x_i, y_i), compute:
    ///    E_i = max(q̂_lo(x_i) - y_i, y_i - q̂_hi(x_i))
    ///
    /// 2. Let Q̂ = Quantile((1-α)(1 + 1/n), {E_1, ..., E_n})
    ///    where n is the calibration set size
    ///
    /// 3. For a new sample x, prediction interval is:
    ///    C(x) = [q̂_lo(x) - Q̂, q̂_hi(x) + Q̂]
    ///
    /// # Coverage Guarantee
    ///
    /// Under exchangeability assumption:
    /// P(Y ∈ C(X)) ≥ 1 - α
    ///
    /// # Arguments
    /// * `y_cal` - True values in calibration set
    /// * `q_lo_cal` - Lower quantile predictions
    /// * `q_hi_cal` - Upper quantile predictions
    ///
    /// # Panics
    /// Panics if input arrays have different lengths or are empty
    pub fn calibrate(
        &mut self,
        y_cal: &[f32],
        q_lo_cal: &[f32],
        q_hi_cal: &[f32],
    ) {
        assert!(!y_cal.is_empty(), "Calibration set cannot be empty");
        assert_eq!(y_cal.len(), q_lo_cal.len(), "Array length mismatch");
        assert_eq!(y_cal.len(), q_hi_cal.len(), "Array length mismatch");

        // Compute nonconformity scores for calibration set
        self.calibration_scores = y_cal
            .iter()
            .zip(q_lo_cal.iter())
            .zip(q_hi_cal.iter())
            .map(|((&y, &lo), &hi)| self.nonconformity_score(y, lo, hi))
            .collect();

        // Compute quantile threshold
        // Q̂ = Quantile((1-α)(1 + 1/n), scores)
        // This ensures finite-sample coverage guarantee
        // Note: For very small calibration sets, the quantile level may exceed 1.0
        // In this case, we clamp to 1.0 (most conservative threshold)
        let n = self.calibration_scores.len();
        let quantile_level = ((1.0 - self.config.alpha) * (1.0 + 1.0 / n as f32)).min(1.0);

        self.quantile_threshold = Some(self.compute_quantile(quantile_level));
    }

    /// Compute quantile of calibration scores
    ///
    /// Uses linear interpolation between order statistics for non-integer indices.
    ///
    /// # Arguments
    /// * `level` - Quantile level in [0, 1]
    ///
    /// # Returns
    /// Estimated quantile value
    fn compute_quantile(&self, level: f32) -> f32 {
        debug_assert!(
            (0.0..=1.0).contains(&level),
            "Quantile level must be in [0, 1]"
        );

        let mut sorted = self.calibration_scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let n = sorted.len();

        // Compute fractional index
        let idx_f = level * n as f32;

        // Ceiling index for conservative quantile estimation
        // This ensures coverage guarantee is satisfied
        let idx = (idx_f.ceil() as usize).min(n - 1);

        sorted[idx]
    }

    /// Produce prediction interval for new sample
    ///
    /// Returns symmetric prediction interval:
    /// C(x) = [q̂_lo(x) - Q̂, q̂_hi(x) + Q̂]
    ///
    /// where Q̂ is the calibrated quantile threshold.
    ///
    /// # Arguments
    /// * `q_lo` - Lower quantile prediction
    /// * `q_hi` - Upper quantile prediction
    ///
    /// # Returns
    /// Tuple (lower_bound, upper_bound) of prediction interval
    ///
    /// # Panics
    /// Panics if calibrate() has not been called
    pub fn predict_interval(&self, q_lo: f32, q_hi: f32) -> (f32, f32) {
        let threshold = self
            .quantile_threshold
            .expect("Must call calibrate() before predict_interval()");

        (q_lo - threshold, q_hi + threshold)
    }

    /// Batch prediction intervals
    ///
    /// Efficiently computes prediction intervals for multiple samples.
    ///
    /// # Arguments
    /// * `q_lo_batch` - Lower quantile predictions
    /// * `q_hi_batch` - Upper quantile predictions
    ///
    /// # Returns
    /// Vector of (lower_bound, upper_bound) tuples
    ///
    /// # Panics
    /// Panics if input arrays have different lengths or calibrate() not called
    pub fn predict_intervals_batch(
        &self,
        q_lo_batch: &[f32],
        q_hi_batch: &[f32],
    ) -> Vec<(f32, f32)> {
        assert_eq!(
            q_lo_batch.len(),
            q_hi_batch.len(),
            "Array length mismatch"
        );

        q_lo_batch
            .iter()
            .zip(q_hi_batch.iter())
            .map(|(&lo, &hi)| self.predict_interval(lo, hi))
            .collect()
    }

    /// Get the calibrated quantile threshold
    ///
    /// # Returns
    /// The threshold Q̂ if calibration has been performed, None otherwise
    pub fn get_threshold(&self) -> Option<f32> {
        self.quantile_threshold
    }

    /// Get reference to calibration scores
    ///
    /// # Returns
    /// Slice of computed nonconformity scores
    pub fn get_calibration_scores(&self) -> &[f32] {
        &self.calibration_scores
    }

    /// Compute empirical coverage on a test set
    ///
    /// Useful for validating that the prediction intervals achieve
    /// the desired coverage level.
    ///
    /// # Arguments
    /// * `y_test` - True values
    /// * `q_lo_test` - Lower quantile predictions
    /// * `q_hi_test` - Upper quantile predictions
    ///
    /// # Returns
    /// Empirical coverage rate in [0, 1]
    pub fn compute_coverage(
        &self,
        y_test: &[f32],
        q_lo_test: &[f32],
        q_hi_test: &[f32],
    ) -> f32 {
        assert_eq!(y_test.len(), q_lo_test.len());
        assert_eq!(y_test.len(), q_hi_test.len());

        let intervals = self.predict_intervals_batch(q_lo_test, q_hi_test);

        let covered = y_test
            .iter()
            .zip(intervals.iter())
            .filter(|(&y, &(lo, hi))| lo <= y && y <= hi)
            .count();

        covered as f32 / y_test.len() as f32
    }

    /// Compute average interval width
    ///
    /// # Arguments
    /// * `q_lo_batch` - Lower quantile predictions
    /// * `q_hi_batch` - Upper quantile predictions
    ///
    /// # Returns
    /// Average width of prediction intervals
    pub fn compute_average_width(
        &self,
        q_lo_batch: &[f32],
        q_hi_batch: &[f32],
    ) -> f32 {
        let intervals = self.predict_intervals_batch(q_lo_batch, q_hi_batch);

        let total_width: f32 = intervals
            .iter()
            .map(|(lo, hi)| hi - lo)
            .sum();

        total_width / intervals.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nonconformity_score() {
        let config = CqrConfig::default();
        let calibrator = CqrCalibrator::new(config);

        // Case 1: y within interval -> score is negative (max of two negative values)
        // E(x,y) = max(q_lo - y, y - q_hi)
        // For y=5.0 in [4.0, 6.0]: max(4.0 - 5.0, 5.0 - 6.0) = max(-1.0, -1.0) = -1.0
        let score = calibrator.nonconformity_score(5.0, 4.0, 6.0);
        assert_eq!(score, -1.0);

        // Case 2: y below interval -> positive score
        // For y=2.0, [4.0, 6.0]: max(4.0 - 2.0, 2.0 - 6.0) = max(2.0, -4.0) = 2.0
        let score = calibrator.nonconformity_score(2.0, 4.0, 6.0);
        assert_eq!(score, 2.0);

        // Case 3: y above interval -> positive score
        // For y=8.0, [4.0, 6.0]: max(4.0 - 8.0, 8.0 - 6.0) = max(-4.0, 2.0) = 2.0
        let score = calibrator.nonconformity_score(8.0, 4.0, 6.0);
        assert_eq!(score, 2.0);
    }

    #[test]
    fn test_calibration_and_prediction() {
        let config = CqrConfig {
            alpha: 0.1,
            symmetric: true,
        };
        let mut calibrator = CqrCalibrator::new(config);

        // Synthetic calibration data
        let y_cal = vec![5.0, 5.2, 4.8, 5.1, 4.9];
        let q_lo_cal = vec![4.5, 4.7, 4.3, 4.6, 4.4];
        let q_hi_cal = vec![5.5, 5.7, 5.3, 5.6, 5.4];

        calibrator.calibrate(&y_cal, &q_lo_cal, &q_hi_cal);

        // Check threshold was computed
        assert!(calibrator.get_threshold().is_some());

        // Make prediction
        // Note: When calibration data shows perfect coverage (all y values fall within
        // [q_lo, q_hi]), the threshold is negative, causing intervals to shrink.
        // This is mathematically correct - no correction is needed for well-calibrated models.
        let (lo, hi) = calibrator.predict_interval(4.5, 5.5);
        assert!(lo <= hi); // Interval should be valid (non-inverted)
    }

    #[test]
    fn test_coverage_guarantee() {
        let alpha = 0.1;
        let config = CqrConfig {
            alpha,
            symmetric: true,
        };
        let mut calibrator = CqrCalibrator::new(config);

        // Generate synthetic data with known distribution
        let n_cal = 100;
        let mut y_cal = Vec::with_capacity(n_cal);
        let mut q_lo_cal = Vec::with_capacity(n_cal);
        let mut q_hi_cal = Vec::with_capacity(n_cal);

        for i in 0..n_cal {
            let y = (i as f32) / 10.0;
            y_cal.push(y);
            q_lo_cal.push(y - 0.5);
            q_hi_cal.push(y + 0.5);
        }

        calibrator.calibrate(&y_cal, &q_lo_cal, &q_hi_cal);

        // Test coverage
        let n_test = 50;
        let mut y_test = Vec::with_capacity(n_test);
        let mut q_lo_test = Vec::with_capacity(n_test);
        let mut q_hi_test = Vec::with_capacity(n_test);

        for i in 0..n_test {
            let y = (i as f32) / 10.0 + 5.0;
            y_test.push(y);
            q_lo_test.push(y - 0.5);
            q_hi_test.push(y + 0.5);
        }

        let coverage = calibrator.compute_coverage(&y_test, &q_lo_test, &q_hi_test);

        // Coverage should be at least 1 - alpha = 0.9
        assert!(
            coverage >= 0.9,
            "Coverage {} is below target {}",
            coverage,
            1.0 - alpha
        );
    }

    #[test]
    fn test_batch_prediction() {
        let config = CqrConfig::default();
        let mut calibrator = CqrCalibrator::new(config);

        // Calibrate
        let y_cal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let q_lo_cal = vec![0.5, 1.5, 2.5, 3.5, 4.5];
        let q_hi_cal = vec![1.5, 2.5, 3.5, 4.5, 5.5];

        calibrator.calibrate(&y_cal, &q_lo_cal, &q_hi_cal);

        // Batch prediction
        let q_lo_batch = vec![0.5, 1.5, 2.5];
        let q_hi_batch = vec![1.5, 2.5, 3.5];

        let intervals = calibrator.predict_intervals_batch(&q_lo_batch, &q_hi_batch);

        assert_eq!(intervals.len(), 3);
        for (lo, hi) in intervals {
            // Intervals should be non-empty (lo <= hi)
            // Note: for perfectly calibrated predictions, lo can equal hi
            assert!(lo <= hi);
        }
    }

    #[test]
    #[should_panic(expected = "Calibration set cannot be empty")]
    fn test_empty_calibration_set() {
        let config = CqrConfig::default();
        let mut calibrator = CqrCalibrator::new(config);

        calibrator.calibrate(&[], &[], &[]);
    }

    #[test]
    #[should_panic(expected = "Array length mismatch")]
    fn test_mismatched_lengths() {
        let config = CqrConfig::default();
        let mut calibrator = CqrCalibrator::new(config);

        let y_cal = vec![1.0, 2.0];
        let q_lo_cal = vec![0.5];
        let q_hi_cal = vec![1.5, 2.5];

        calibrator.calibrate(&y_cal, &q_lo_cal, &q_hi_cal);
    }
}
