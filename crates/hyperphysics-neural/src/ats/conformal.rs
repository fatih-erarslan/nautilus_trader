//! Conformal Prediction Bridge for HFT
//!
//! Wraps ats-core's conformal prediction with FANN-specific optimizations.

use std::collections::VecDeque;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::error::{NeuralError, NeuralResult};

/// Configuration for conformal prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformalConfig {
    /// Target confidence level (e.g., 0.95 for 95%)
    pub confidence: f64,
    /// Minimum calibration samples required
    pub min_calibration_size: usize,
    /// Maximum calibration window size
    pub max_calibration_size: usize,
    /// Target latency in microseconds
    pub target_latency_us: u64,
    /// Use online calibration updates
    pub online_calibration: bool,
}

impl Default for ConformalConfig {
    fn default() -> Self {
        Self {
            confidence: 0.95,
            min_calibration_size: 100,
            max_calibration_size: 10000,
            target_latency_us: 20, // 20μs target for HFT
            online_calibration: true,
        }
    }
}

impl ConformalConfig {
    /// HFT-optimized configuration
    pub fn hft() -> Self {
        Self {
            confidence: 0.95,
            min_calibration_size: 50,
            max_calibration_size: 1000,
            target_latency_us: 10, // 10μs for HFT
            online_calibration: true,
        }
    }

    /// High-coverage configuration
    pub fn high_coverage() -> Self {
        Self {
            confidence: 0.99,
            min_calibration_size: 500,
            max_calibration_size: 50000,
            target_latency_us: 100,
            online_calibration: true,
        }
    }
}

/// Uncertainty bounds for a prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyBounds {
    /// Point prediction
    pub prediction: f64,
    /// Lower bound of prediction interval
    pub lower: f64,
    /// Upper bound of prediction interval
    pub upper: f64,
    /// Confidence level
    pub confidence: f64,
    /// Interval width (uncertainty measure)
    pub width: f64,
    /// Computation time in nanoseconds
    pub latency_ns: u64,
}

impl UncertaintyBounds {
    /// Check if a value falls within bounds
    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }

    /// Relative uncertainty (width / prediction)
    pub fn relative_uncertainty(&self) -> f64 {
        if self.prediction.abs() > 1e-10 {
            self.width / self.prediction.abs()
        } else {
            f64::INFINITY
        }
    }
}

/// Fast conformal predictor optimized for FANN integration
pub struct FastConformalPredictor {
    /// Configuration
    config: ConformalConfig,
    /// Rolling calibration scores
    calibration_buffer: VecDeque<f64>,
    /// Pre-sorted calibration for fast quantile
    sorted_calibration: Vec<f64>,
    /// Cached quantile threshold
    cached_quantile: Option<f64>,
    /// Statistics
    total_predictions: u64,
    total_latency_ns: u64,
}

impl FastConformalPredictor {
    /// Create new predictor with configuration
    pub fn new(config: ConformalConfig) -> Self {
        Self {
            calibration_buffer: VecDeque::with_capacity(config.max_calibration_size),
            sorted_calibration: Vec::with_capacity(config.max_calibration_size),
            cached_quantile: None,
            config,
            total_predictions: 0,
            total_latency_ns: 0,
        }
    }

    /// Create HFT-optimized predictor
    pub fn hft() -> Self {
        Self::new(ConformalConfig::hft())
    }

    /// Add calibration score (nonconformity score)
    pub fn add_calibration_score(&mut self, score: f64) {
        // Maintain bounded buffer
        if self.calibration_buffer.len() >= self.config.max_calibration_size {
            self.calibration_buffer.pop_front();
        }
        self.calibration_buffer.push_back(score);

        // Invalidate cache when calibration changes
        self.cached_quantile = None;
    }

    /// Batch add calibration scores
    pub fn add_calibration_batch(&mut self, scores: &[f64]) {
        for &score in scores {
            self.add_calibration_score(score);
        }
    }

    /// Compute calibration score for prediction vs actual
    pub fn compute_calibration_score(prediction: f64, actual: f64) -> f64 {
        (prediction - actual).abs()
    }

    /// Get prediction interval for a point prediction
    pub fn predict(&mut self, prediction: f64) -> NeuralResult<UncertaintyBounds> {
        let start = Instant::now();

        if self.calibration_buffer.len() < self.config.min_calibration_size {
            return Err(NeuralError::InvalidInput(format!(
                "Need at least {} calibration samples, have {}",
                self.config.min_calibration_size,
                self.calibration_buffer.len()
            )));
        }

        // Get quantile threshold (cached or computed)
        let quantile = self.get_quantile_threshold()?;

        let latency_ns = start.elapsed().as_nanos() as u64;
        self.total_predictions += 1;
        self.total_latency_ns += latency_ns;

        Ok(UncertaintyBounds {
            prediction,
            lower: prediction - quantile,
            upper: prediction + quantile,
            confidence: self.config.confidence,
            width: 2.0 * quantile,
            latency_ns,
        })
    }

    /// Batch prediction intervals
    pub fn predict_batch(&mut self, predictions: &[f64]) -> NeuralResult<Vec<UncertaintyBounds>> {
        // Compute quantile once for all predictions
        let quantile = self.get_quantile_threshold()?;
        let start = Instant::now();

        let results: Vec<_> = predictions
            .iter()
            .map(|&pred| UncertaintyBounds {
                prediction: pred,
                lower: pred - quantile,
                upper: pred + quantile,
                confidence: self.config.confidence,
                width: 2.0 * quantile,
                latency_ns: 0,
            })
            .collect();

        let total_latency = start.elapsed().as_nanos() as u64;
        let per_pred_latency = total_latency / predictions.len().max(1) as u64;

        // Update individual latencies
        Ok(results
            .into_iter()
            .map(|mut b| {
                b.latency_ns = per_pred_latency;
                b
            })
            .collect())
    }

    /// Get quantile threshold (with caching)
    fn get_quantile_threshold(&mut self) -> NeuralResult<f64> {
        if let Some(cached) = self.cached_quantile {
            return Ok(cached);
        }

        // Sort calibration scores
        self.sorted_calibration = self.calibration_buffer.iter().copied().collect();
        self.sorted_calibration
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Compute quantile index
        let n = self.sorted_calibration.len();
        let idx = ((self.config.confidence * (n + 1) as f64).ceil() as usize).min(n) - 1;

        let quantile = self.sorted_calibration[idx];
        self.cached_quantile = Some(quantile);

        Ok(quantile)
    }

    /// Update with new observation (online calibration)
    pub fn update(&mut self, prediction: f64, actual: f64) {
        if self.config.online_calibration {
            let score = Self::compute_calibration_score(prediction, actual);
            self.add_calibration_score(score);
        }
    }

    /// Get average latency in nanoseconds
    pub fn avg_latency_ns(&self) -> f64 {
        if self.total_predictions == 0 {
            0.0
        } else {
            self.total_latency_ns as f64 / self.total_predictions as f64
        }
    }

    /// Get calibration buffer size
    pub fn calibration_size(&self) -> usize {
        self.calibration_buffer.len()
    }

    /// Clear calibration data
    pub fn clear(&mut self) {
        self.calibration_buffer.clear();
        self.sorted_calibration.clear();
        self.cached_quantile = None;
    }

    /// Get configuration
    pub fn config(&self) -> &ConformalConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conformal_predictor_creation() {
        let predictor = FastConformalPredictor::hft();
        assert_eq!(predictor.config.target_latency_us, 10);
        assert_eq!(predictor.calibration_size(), 0);
    }

    #[test]
    fn test_add_calibration_scores() {
        let mut predictor = FastConformalPredictor::new(ConformalConfig {
            min_calibration_size: 5,
            max_calibration_size: 10,
            ..Default::default()
        });

        for i in 0..15 {
            predictor.add_calibration_score(i as f64);
        }

        // Should be capped at max size
        assert_eq!(predictor.calibration_size(), 10);
    }

    #[test]
    fn test_prediction_interval() {
        let mut predictor = FastConformalPredictor::new(ConformalConfig {
            min_calibration_size: 5,
            confidence: 0.95,
            ..Default::default()
        });

        // Add calibration scores
        for i in 1..=100 {
            predictor.add_calibration_score(i as f64 * 0.1);
        }

        let bounds = predictor.predict(50.0).unwrap();

        assert!(bounds.lower < 50.0);
        assert!(bounds.upper > 50.0);
        assert_eq!(bounds.confidence, 0.95);
        assert!(bounds.width > 0.0);
    }

    #[test]
    fn test_batch_prediction() {
        let mut predictor = FastConformalPredictor::new(ConformalConfig {
            min_calibration_size: 10,
            ..Default::default()
        });

        for i in 1..=100 {
            predictor.add_calibration_score(i as f64 * 0.05);
        }

        let predictions = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let bounds = predictor.predict_batch(&predictions).unwrap();

        assert_eq!(bounds.len(), 5);
        for (b, &p) in bounds.iter().zip(predictions.iter()) {
            assert_eq!(b.prediction, p);
            assert!(b.lower < p);
            assert!(b.upper > p);
        }
    }

    #[test]
    fn test_online_calibration() {
        let mut predictor = FastConformalPredictor::new(ConformalConfig {
            min_calibration_size: 5,
            online_calibration: true,
            ..Default::default()
        });

        // Simulate predictions and actuals
        for i in 0..20 {
            let pred = i as f64;
            let actual = pred + (i as f64 * 0.1);
            predictor.update(pred, actual);
        }

        assert!(predictor.calibration_size() >= 5);
    }

    #[test]
    fn test_uncertainty_bounds_contains() {
        let bounds = UncertaintyBounds {
            prediction: 100.0,
            lower: 90.0,
            upper: 110.0,
            confidence: 0.95,
            width: 20.0,
            latency_ns: 100,
        };

        assert!(bounds.contains(100.0));
        assert!(bounds.contains(95.0));
        assert!(!bounds.contains(85.0));
        assert!(!bounds.contains(115.0));
    }

    #[test]
    fn test_relative_uncertainty() {
        let bounds = UncertaintyBounds {
            prediction: 100.0,
            lower: 90.0,
            upper: 110.0,
            confidence: 0.95,
            width: 20.0,
            latency_ns: 100,
        };

        assert!((bounds.relative_uncertainty() - 0.2).abs() < 0.001);
    }
}
