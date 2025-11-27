//! Symmetric CQR with Enhanced Diagnostics
//!
//! This module extends the base CQR implementation with additional diagnostic
//! capabilities and symmetric interval guarantees.

use super::base::{CqrCalibrator, CqrConfig};

/// Symmetric CQR with diagnostic utilities
///
/// Wraps the base CQR calibrator with additional analysis capabilities
/// for monitoring prediction interval quality.
pub struct SymmetricCqr {
    calibrator: CqrCalibrator,
}

impl SymmetricCqr {
    /// Create new symmetric CQR
    pub fn new(config: CqrConfig) -> Self {
        Self {
            calibrator: CqrCalibrator::new(config),
        }
    }

    /// Calibrate the model
    pub fn calibrate(
        &mut self,
        y_cal: &[f32],
        q_lo_cal: &[f32],
        q_hi_cal: &[f32],
    ) {
        self.calibrator.calibrate(y_cal, q_lo_cal, q_hi_cal);
    }

    /// Get prediction interval
    pub fn predict_interval(&self, q_lo: f32, q_hi: f32) -> (f32, f32) {
        self.calibrator.predict_interval(q_lo, q_hi)
    }

    /// Batch predictions
    pub fn predict_intervals_batch(
        &self,
        q_lo_batch: &[f32],
        q_hi_batch: &[f32],
    ) -> Vec<(f32, f32)> {
        self.calibrator.predict_intervals_batch(q_lo_batch, q_hi_batch)
    }

    /// Compute interval statistics
    ///
    /// Returns detailed statistics about prediction intervals:
    /// - mean_width: Average interval width
    /// - median_width: Median interval width
    /// - min_width: Minimum interval width
    /// - max_width: Maximum interval width
    /// - std_width: Standard deviation of widths
    pub fn compute_interval_statistics(
        &self,
        q_lo_batch: &[f32],
        q_hi_batch: &[f32],
    ) -> IntervalStatistics {
        let intervals = self.predict_intervals_batch(q_lo_batch, q_hi_batch);

        let widths: Vec<f32> = intervals
            .iter()
            .map(|(lo, hi)| hi - lo)
            .collect();

        let mean_width = widths.iter().sum::<f32>() / widths.len() as f32;

        let mut sorted_widths = widths.clone();
        sorted_widths.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_width = sorted_widths[sorted_widths.len() / 2];

        let min_width = sorted_widths[0];
        let max_width = sorted_widths[sorted_widths.len() - 1];

        let variance = widths
            .iter()
            .map(|w| (w - mean_width).powi(2))
            .sum::<f32>() / widths.len() as f32;
        let std_width = variance.sqrt();

        IntervalStatistics {
            mean_width,
            median_width,
            min_width,
            max_width,
            std_width,
        }
    }

    /// Compute coverage and efficiency metrics
    ///
    /// Analyzes prediction intervals on test set:
    /// - coverage: Empirical coverage rate
    /// - average_width: Mean interval width
    /// - efficiency: Coverage / average_width (higher is better)
    pub fn evaluate(
        &self,
        y_test: &[f32],
        q_lo_test: &[f32],
        q_hi_test: &[f32],
    ) -> EvaluationMetrics {
        let coverage = self.calibrator.compute_coverage(y_test, q_lo_test, q_hi_test);
        let average_width = self.calibrator.compute_average_width(q_lo_test, q_hi_test);
        let efficiency = coverage / average_width;

        EvaluationMetrics {
            coverage,
            average_width,
            efficiency,
        }
    }
}

/// Statistics about prediction interval widths
#[derive(Debug, Clone)]
pub struct IntervalStatistics {
    pub mean_width: f32,
    pub median_width: f32,
    pub min_width: f32,
    pub max_width: f32,
    pub std_width: f32,
}

/// Evaluation metrics for CQR performance
#[derive(Debug, Clone)]
pub struct EvaluationMetrics {
    /// Empirical coverage rate
    pub coverage: f32,
    /// Average prediction interval width
    pub average_width: f32,
    /// Efficiency score (coverage per unit width)
    pub efficiency: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_cqr_workflow() {
        let config = CqrConfig::default();
        let mut cqr = SymmetricCqr::new(config);

        // Calibration
        let y_cal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let q_lo_cal = vec![0.5, 1.5, 2.5, 3.5, 4.5];
        let q_hi_cal = vec![1.5, 2.5, 3.5, 4.5, 5.5];

        cqr.calibrate(&y_cal, &q_lo_cal, &q_hi_cal);

        // Prediction
        let (lo, hi) = cqr.predict_interval(1.0, 2.0);
        assert!(lo < hi);
    }

    #[test]
    fn test_interval_statistics() {
        let config = CqrConfig::default();
        let mut cqr = SymmetricCqr::new(config);

        let y_cal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let q_lo_cal = vec![0.5, 1.5, 2.5, 3.5, 4.5];
        let q_hi_cal = vec![1.5, 2.5, 3.5, 4.5, 5.5];

        cqr.calibrate(&y_cal, &q_lo_cal, &q_hi_cal);

        let stats = cqr.compute_interval_statistics(&q_lo_cal, &q_hi_cal);

        assert!(stats.mean_width > 0.0);
        assert!(stats.min_width <= stats.median_width);
        assert!(stats.median_width <= stats.max_width);
    }

    #[test]
    fn test_evaluation_metrics() {
        let config = CqrConfig::default();
        let mut cqr = SymmetricCqr::new(config);

        let y_cal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let q_lo_cal = vec![0.5, 1.5, 2.5, 3.5, 4.5];
        let q_hi_cal = vec![1.5, 2.5, 3.5, 4.5, 5.5];

        cqr.calibrate(&y_cal, &q_lo_cal, &q_hi_cal);

        let y_test = vec![1.5, 2.5, 3.5];
        let q_lo_test = vec![1.0, 2.0, 3.0];
        let q_hi_test = vec![2.0, 3.0, 4.0];

        let metrics = cqr.evaluate(&y_test, &q_lo_test, &q_hi_test);

        assert!(metrics.coverage >= 0.0 && metrics.coverage <= 1.0);
        assert!(metrics.average_width > 0.0);
        assert!(metrics.efficiency > 0.0);
    }
}
