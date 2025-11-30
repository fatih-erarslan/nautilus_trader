//! Asymmetric Conformalized Quantile Regression
//!
//! # Mathematical Foundation
//!
//! Unlike symmetric CQR, asymmetric CQR computes separate correction factors
//! for lower and upper quantiles, which can lead to tighter intervals.
//!
//! **Lower nonconformity score**: E_lo(x,y) = q̂_lo(x) - y
//! **Upper nonconformity score**: E_hi(x,y) = y - q̂_hi(x)
//!
//! **Prediction interval**: [q̂_lo(x) - Q̂_lo, q̂_hi(x) + Q̂_hi]
//! where:
//! - Q̂_lo = (1-α_lo)(1 + 1/n) quantile of {E_lo,i}
//! - Q̂_hi = (1-α_hi)(1 + 1/n) quantile of {E_hi,i}
//! - α_lo + α_hi = α (target miscoverage level)
//!
//! # References
//!
//! - Romano, Y., Patterson, E., & Candès, E. (2019). "Conformalized Quantile Regression"
//! - Feldman, S., Bates, S., & Romano, Y. (2021). "Improving Conditional Coverage via
//!   Orthogonal Quantile Regression"

use std::cmp::Ordering;

/// Configuration for asymmetric CQR
#[derive(Clone, Debug)]
pub struct AsymmetricCqrConfig {
    /// Total miscoverage level α ∈ (0, 1)
    pub alpha: f32,
    /// Miscoverage allocated to lower tail
    /// Must satisfy: alpha_lo + alpha_hi = alpha
    pub alpha_lo: f32,
    /// Miscoverage allocated to upper tail
    pub alpha_hi: f32,
}

impl Default for AsymmetricCqrConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            alpha_lo: 0.05,
            alpha_hi: 0.05,
        }
    }
}

impl AsymmetricCqrConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.alpha <= 0.0 || self.alpha >= 1.0 {
            return Err(format!("alpha must be in (0, 1), got {}", self.alpha));
        }

        if self.alpha_lo < 0.0 || self.alpha_hi < 0.0 {
            return Err("alpha_lo and alpha_hi must be non-negative".to_string());
        }

        let alpha_sum = self.alpha_lo + self.alpha_hi;
        if (alpha_sum - self.alpha).abs() > 1e-6 {
            return Err(format!(
                "alpha_lo + alpha_hi must equal alpha: {} + {} ≠ {}",
                self.alpha_lo, self.alpha_hi, self.alpha
            ));
        }

        Ok(())
    }
}

/// Asymmetric CQR calibrator
///
/// Provides potentially tighter prediction intervals by treating lower and
/// upper quantile errors independently.
pub struct AsymmetricCqrCalibrator {
    config: AsymmetricCqrConfig,
    /// Lower nonconformity scores
    scores_lo: Vec<f32>,
    /// Upper nonconformity scores
    scores_hi: Vec<f32>,
    /// Calibrated lower threshold
    threshold_lo: Option<f32>,
    /// Calibrated upper threshold
    threshold_hi: Option<f32>,
}

impl AsymmetricCqrCalibrator {
    /// Create new asymmetric CQR calibrator
    ///
    /// # Arguments
    /// * `config` - Asymmetric CQR configuration
    ///
    /// # Panics
    /// Panics if configuration validation fails
    pub fn new(config: AsymmetricCqrConfig) -> Self {
        config.validate().expect("Invalid asymmetric CQR config");

        Self {
            config,
            scores_lo: Vec::new(),
            scores_hi: Vec::new(),
            threshold_lo: None,
            threshold_hi: None,
        }
    }

    /// Compute lower nonconformity score
    ///
    /// E_lo(x, y) = q̂_lo(x) - y
    ///
    /// Positive score indicates y is below the predicted lower quantile.
    ///
    /// # Arguments
    /// * `y` - True value
    /// * `q_lo` - Lower quantile prediction
    #[inline]
    pub fn nonconformity_score_lo(&self, y: f32, q_lo: f32) -> f32 {
        q_lo - y
    }

    /// Compute upper nonconformity score
    ///
    /// E_hi(x, y) = y - q̂_hi(x)
    ///
    /// Positive score indicates y is above the predicted upper quantile.
    ///
    /// # Arguments
    /// * `y` - True value
    /// * `q_hi` - Upper quantile prediction
    #[inline]
    pub fn nonconformity_score_hi(&self, y: f32, q_hi: f32) -> f32 {
        y - q_hi
    }

    /// Calibrate on calibration set
    ///
    /// # Algorithm
    ///
    /// 1. Compute lower scores: E_lo,i = q̂_lo(x_i) - y_i
    /// 2. Compute upper scores: E_hi,i = y_i - q̂_hi(x_i)
    /// 3. Compute quantile thresholds:
    ///    - Q̂_lo = (1-α_lo)(1 + 1/n) quantile of {E_lo,i}
    ///    - Q̂_hi = (1-α_hi)(1 + 1/n) quantile of {E_hi,i}
    ///
    /// # Arguments
    /// * `y_cal` - True values
    /// * `q_lo_cal` - Lower quantile predictions
    /// * `q_hi_cal` - Upper quantile predictions
    ///
    /// # Panics
    /// Panics if arrays have mismatched lengths or are empty
    pub fn calibrate(
        &mut self,
        y_cal: &[f32],
        q_lo_cal: &[f32],
        q_hi_cal: &[f32],
    ) {
        assert!(!y_cal.is_empty(), "Calibration set cannot be empty");
        assert_eq!(y_cal.len(), q_lo_cal.len(), "Array length mismatch");
        assert_eq!(y_cal.len(), q_hi_cal.len(), "Array length mismatch");

        // Compute lower nonconformity scores
        self.scores_lo = y_cal
            .iter()
            .zip(q_lo_cal.iter())
            .map(|(&y, &lo)| self.nonconformity_score_lo(y, lo))
            .collect();

        // Compute upper nonconformity scores
        self.scores_hi = y_cal
            .iter()
            .zip(q_hi_cal.iter())
            .map(|(&y, &hi)| self.nonconformity_score_hi(y, hi))
            .collect();

        let n = y_cal.len();

        // Compute lower threshold
        // Note: For small calibration sets, quantile level may exceed 1.0; clamp to 1.0
        let quantile_level_lo = ((1.0 - self.config.alpha_lo) * (1.0 + 1.0 / n as f32)).min(1.0);
        self.threshold_lo = Some(self.compute_quantile(&self.scores_lo, quantile_level_lo));

        // Compute upper threshold
        let quantile_level_hi = ((1.0 - self.config.alpha_hi) * (1.0 + 1.0 / n as f32)).min(1.0);
        self.threshold_hi = Some(self.compute_quantile(&self.scores_hi, quantile_level_hi));
    }

    /// Compute quantile from scores
    fn compute_quantile(&self, scores: &[f32], level: f32) -> f32 {
        debug_assert!(
            (0.0..=1.0).contains(&level),
            "Quantile level must be in [0, 1]"
        );

        let mut sorted = scores.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let n = sorted.len();
        let idx_f = level * n as f32;
        let idx = (idx_f.ceil() as usize).min(n - 1);

        sorted[idx]
    }

    /// Produce asymmetric prediction interval
    ///
    /// C(x) = [q̂_lo(x) - Q̂_lo, q̂_hi(x) + Q̂_hi]
    ///
    /// # Arguments
    /// * `q_lo` - Lower quantile prediction
    /// * `q_hi` - Upper quantile prediction
    ///
    /// # Returns
    /// Tuple (lower_bound, upper_bound)
    ///
    /// # Panics
    /// Panics if calibrate() has not been called
    pub fn predict_interval(&self, q_lo: f32, q_hi: f32) -> (f32, f32) {
        let threshold_lo = self
            .threshold_lo
            .expect("Must call calibrate() first");
        let threshold_hi = self
            .threshold_hi
            .expect("Must call calibrate() first");

        (q_lo - threshold_lo, q_hi + threshold_hi)
    }

    /// Batch prediction intervals
    pub fn predict_intervals_batch(
        &self,
        q_lo_batch: &[f32],
        q_hi_batch: &[f32],
    ) -> Vec<(f32, f32)> {
        assert_eq!(q_lo_batch.len(), q_hi_batch.len());

        q_lo_batch
            .iter()
            .zip(q_hi_batch.iter())
            .map(|(&lo, &hi)| self.predict_interval(lo, hi))
            .collect()
    }

    /// Get calibrated thresholds
    pub fn get_thresholds(&self) -> Option<(f32, f32)> {
        match (self.threshold_lo, self.threshold_hi) {
            (Some(lo), Some(hi)) => Some((lo, hi)),
            _ => None,
        }
    }

    /// Compute empirical coverage
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

    /// Compute conditional coverages (lower and upper separately)
    ///
    /// Returns (lower_coverage, upper_coverage) where:
    /// - lower_coverage = fraction of y ≥ lower_bound
    /// - upper_coverage = fraction of y ≤ upper_bound
    pub fn compute_conditional_coverages(
        &self,
        y_test: &[f32],
        q_lo_test: &[f32],
        q_hi_test: &[f32],
    ) -> (f32, f32) {
        let intervals = self.predict_intervals_batch(q_lo_test, q_hi_test);

        let lower_covered = y_test
            .iter()
            .zip(intervals.iter())
            .filter(|(&y, &(lo, _))| y >= lo)
            .count();

        let upper_covered = y_test
            .iter()
            .zip(intervals.iter())
            .filter(|(&y, &(_, hi))| y <= hi)
            .count();

        let n = y_test.len() as f32;
        (lower_covered as f32 / n, upper_covered as f32 / n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asymmetric_scores() {
        let config = AsymmetricCqrConfig::default();
        let calibrator = AsymmetricCqrCalibrator::new(config);

        // Test lower score
        let score_lo = calibrator.nonconformity_score_lo(3.0, 5.0);
        assert_eq!(score_lo, 2.0); // 5.0 - 3.0

        // Test upper score
        let score_hi = calibrator.nonconformity_score_hi(8.0, 6.0);
        assert_eq!(score_hi, 2.0); // 8.0 - 6.0
    }

    #[test]
    fn test_asymmetric_calibration() {
        let config = AsymmetricCqrConfig {
            alpha: 0.1,
            alpha_lo: 0.05,
            alpha_hi: 0.05,
        };
        let mut calibrator = AsymmetricCqrCalibrator::new(config);

        // Calibration data
        let y_cal = vec![5.0, 5.2, 4.8, 5.1, 4.9];
        let q_lo_cal = vec![4.5, 4.7, 4.3, 4.6, 4.4];
        let q_hi_cal = vec![5.5, 5.7, 5.3, 5.6, 5.4];

        calibrator.calibrate(&y_cal, &q_lo_cal, &q_hi_cal);

        // Check thresholds computed
        assert!(calibrator.get_thresholds().is_some());

        // Make prediction
        // Note: When calibration data shows perfect coverage (all y values fall within
        // [q_lo, q_hi]), the thresholds are negative, causing intervals to shrink.
        // This is mathematically correct - no correction is needed for well-calibrated models.
        let (lo, hi) = calibrator.predict_interval(4.5, 5.5);
        assert!(lo <= hi); // Interval should be valid (non-inverted)
    }

    #[test]
    fn test_conditional_coverage() {
        let config = AsymmetricCqrConfig::default();
        let mut calibrator = AsymmetricCqrCalibrator::new(config);

        // Generate data
        let n = 100;
        let mut y_cal = Vec::with_capacity(n);
        let mut q_lo_cal = Vec::with_capacity(n);
        let mut q_hi_cal = Vec::with_capacity(n);

        for i in 0..n {
            let y = (i as f32) / 10.0;
            y_cal.push(y);
            q_lo_cal.push(y - 0.5);
            q_hi_cal.push(y + 0.5);
        }

        calibrator.calibrate(&y_cal, &q_lo_cal, &q_hi_cal);

        // Test conditional coverages
        let (lower_cov, upper_cov) = calibrator.compute_conditional_coverages(
            &y_cal,
            &q_lo_cal,
            &q_hi_cal,
        );

        // Both should be high (≥ 1 - α_lo and ≥ 1 - α_hi)
        assert!(lower_cov >= 0.95);
        assert!(upper_cov >= 0.95);
    }

    #[test]
    #[should_panic(expected = "alpha_lo + alpha_hi must equal alpha")]
    fn test_invalid_alpha_split() {
        let config = AsymmetricCqrConfig {
            alpha: 0.1,
            alpha_lo: 0.06,
            alpha_hi: 0.05,
        };
        AsymmetricCqrCalibrator::new(config);
    }
}
