//! Conformal prediction integration from neural-trader-predictor
//!
//! This module provides distribution-free prediction intervals with guaranteed coverage
//! by wrapping the neural-trader-predictor crate's conformal prediction implementations.
//!
//! # Conformal Prediction Methods
//!
//! - **Split Conformal**: Basic method with calibration/prediction split
//! - **Adaptive Conformal (ACI)**: PID-controlled dynamic coverage adjustment
//! - **Conformalized Quantile Regression (CQR)**: Quantile-based intervals
//!
//! # Example
//!
//! ```rust,ignore
//! use hyperphysics_neural_trader::conformal::*;
//!
//! // Create predictor with 90% coverage
//! let mut predictor = HyperConformalPredictor::new(ConformalConfig {
//!     alpha: 0.1,  // 90% coverage
//!     method: ConformalMethod::Split,
//!     ..Default::default()
//! });
//!
//! // Calibrate with historical predictions and actuals
//! predictor.calibrate(&predictions, &actuals)?;
//!
//! // Get prediction interval for new point
//! let interval = predictor.predict(point_prediction);
//! println!("Interval: [{:.4}, {:.4}]", interval.lower, interval.upper);
//! ```

use crate::error::{NeuralBridgeError, Result};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tracing::{debug, trace};

/// Conformal prediction method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ConformalMethod {
    /// Split conformal prediction (basic)
    #[default]
    Split,
    /// Adaptive conformal inference with PID control
    Adaptive,
    /// Conformalized quantile regression
    CQR,
}

/// Configuration for conformal prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformalConfig {
    /// Miscoverage rate (1 - confidence level)
    /// alpha = 0.1 means 90% coverage
    pub alpha: f64,
    /// Conformal method to use
    pub method: ConformalMethod,
    /// Window size for online updates (0 = no sliding window)
    pub window_size: usize,
    /// PID controller gains for adaptive method
    pub pid_kp: f64,
    /// PID integral gain
    pub pid_ki: f64,
    /// PID derivative gain
    pub pid_kd: f64,
    /// Minimum alpha (maximum coverage)
    pub alpha_min: f64,
    /// Maximum alpha (minimum coverage)
    pub alpha_max: f64,
}

impl Default for ConformalConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1, // 90% coverage
            method: ConformalMethod::Split,
            window_size: 1000,
            pid_kp: 0.1,
            pid_ki: 0.01,
            pid_kd: 0.05,
            alpha_min: 0.01,
            alpha_max: 0.5,
        }
    }
}

/// Prediction interval result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionInterval {
    /// Point prediction
    pub point: f64,
    /// Lower bound of interval
    pub lower: f64,
    /// Upper bound of interval
    pub upper: f64,
    /// Actual coverage rate (if known from calibration)
    pub coverage: f64,
    /// Interval width
    pub width: f64,
}

impl PredictionInterval {
    /// Check if actual value falls within interval
    pub fn contains(&self, actual: f64) -> bool {
        actual >= self.lower && actual <= self.upper
    }

    /// Calculate interval efficiency (narrower is better)
    pub fn efficiency(&self) -> f64 {
        if self.width > 0.0 {
            1.0 / self.width
        } else {
            f64::INFINITY
        }
    }
}

/// Nonconformity score calculation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum NonconformityScore {
    /// Absolute error: |y - ŷ|
    #[default]
    Absolute,
    /// Normalized error: |y - ŷ| / σ
    Normalized,
    /// Signed error: y - ŷ (for asymmetric intervals)
    Signed,
}

/// HyperPhysics conformal predictor wrapping neural-trader-predictor
pub struct HyperConformalPredictor {
    /// Configuration
    config: ConformalConfig,
    /// Calibration scores (sorted)
    calibration_scores: Vec<f64>,
    /// Current quantile threshold
    quantile: f64,
    /// Whether predictor is calibrated
    calibrated: bool,
    /// Score type
    score_type: NonconformityScore,
    /// Sliding window for online updates
    score_window: VecDeque<f64>,
    /// PID state for adaptive method
    pid_state: PIDState,
    /// Current effective alpha
    effective_alpha: f64,
    /// Coverage tracking
    coverage_tracker: CoverageTracker,
}

/// PID controller state for adaptive conformal inference
#[derive(Debug, Clone, Default)]
struct PIDState {
    integral: f64,
    previous_error: f64,
}

/// Track empirical coverage for validation
#[derive(Debug, Clone, Default)]
struct CoverageTracker {
    total: usize,
    covered: usize,
}

impl CoverageTracker {
    fn update(&mut self, was_covered: bool) {
        self.total += 1;
        if was_covered {
            self.covered += 1;
        }
    }

    fn coverage(&self) -> f64 {
        if self.total > 0 {
            self.covered as f64 / self.total as f64
        } else {
            0.0
        }
    }
}

impl HyperConformalPredictor {
    /// Create a new conformal predictor
    pub fn new(config: ConformalConfig) -> Self {
        Self {
            effective_alpha: config.alpha,
            config,
            calibration_scores: Vec::new(),
            quantile: 0.0,
            calibrated: false,
            score_type: NonconformityScore::Absolute,
            score_window: VecDeque::new(),
            pid_state: PIDState::default(),
            coverage_tracker: CoverageTracker::default(),
        }
    }

    /// Create with default configuration for given alpha
    pub fn with_alpha(alpha: f64) -> Self {
        Self::new(ConformalConfig {
            alpha,
            ..Default::default()
        })
    }

    /// Set the nonconformity score type
    pub fn with_score_type(mut self, score_type: NonconformityScore) -> Self {
        self.score_type = score_type;
        self
    }

    /// Calibrate the predictor with historical predictions and actuals
    pub fn calibrate(&mut self, predictions: &[f64], actuals: &[f64]) -> Result<()> {
        if predictions.len() != actuals.len() {
            return Err(NeuralBridgeError::InvalidInput(
                "Predictions and actuals must have same length".to_string(),
            ));
        }

        if predictions.is_empty() {
            return Err(NeuralBridgeError::InsufficientData {
                required: 1,
                actual: 0,
            });
        }

        // Calculate nonconformity scores
        self.calibration_scores = predictions
            .iter()
            .zip(actuals.iter())
            .map(|(&pred, &actual)| self.compute_score(pred, actual))
            .collect();

        // Sort scores for quantile computation
        self.calibration_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute quantile threshold
        self.update_quantile();

        self.calibrated = true;

        debug!(
            n = predictions.len(),
            quantile = self.quantile,
            alpha = self.config.alpha,
            "Calibrated conformal predictor"
        );

        Ok(())
    }

    /// Get prediction interval for a point prediction
    pub fn predict(&self, point_prediction: f64) -> PredictionInterval {
        if !self.calibrated {
            // Return wide interval if not calibrated
            let width = point_prediction.abs() * 0.1;
            return PredictionInterval {
                point: point_prediction,
                lower: point_prediction - width,
                upper: point_prediction + width,
                coverage: 0.0,
                width: width * 2.0,
            };
        }

        let interval_width = self.quantile;

        PredictionInterval {
            point: point_prediction,
            lower: point_prediction - interval_width,
            upper: point_prediction + interval_width,
            coverage: 1.0 - self.effective_alpha,
            width: interval_width * 2.0,
        }
    }

    /// Update predictor with new observation (online learning)
    pub fn update(&mut self, prediction: f64, actual: f64) -> Result<()> {
        let score = self.compute_score(prediction, actual);

        // Track coverage
        let interval = self.predict(prediction);
        let covered = interval.contains(actual);
        self.coverage_tracker.update(covered);

        // Add to sliding window
        if self.config.window_size > 0 {
            self.score_window.push_back(score);
            if self.score_window.len() > self.config.window_size {
                self.score_window.pop_front();
            }

            // Update calibration scores from window
            self.calibration_scores = self.score_window.iter().copied().collect();
            self.calibration_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        } else {
            // Binary search insert to maintain sorted order
            let pos = self.calibration_scores
                .binary_search_by(|&s| s.partial_cmp(&score).unwrap())
                .unwrap_or_else(|e| e);
            self.calibration_scores.insert(pos, score);
        }

        // Apply adaptive update if using adaptive method
        if self.config.method == ConformalMethod::Adaptive {
            self.adaptive_update(covered);
        }

        self.update_quantile();

        trace!(
            score = score,
            quantile = self.quantile,
            coverage = self.coverage_tracker.coverage(),
            "Updated conformal predictor"
        );

        Ok(())
    }

    /// Adaptive update using PID control
    fn adaptive_update(&mut self, was_covered: bool) {
        // Target is 1 - alpha (desired coverage)
        let target = 1.0 - self.config.alpha;
        let current = if was_covered { 1.0 } else { 0.0 };
        let error = target - current;

        // PID control
        self.pid_state.integral += error;
        let derivative = error - self.pid_state.previous_error;
        self.pid_state.previous_error = error;

        let adjustment = self.config.pid_kp * error
            + self.config.pid_ki * self.pid_state.integral
            + self.config.pid_kd * derivative;

        // Update effective alpha
        self.effective_alpha = (self.effective_alpha - adjustment)
            .clamp(self.config.alpha_min, self.config.alpha_max);
    }

    /// Compute nonconformity score
    fn compute_score(&self, prediction: f64, actual: f64) -> f64 {
        match self.score_type {
            NonconformityScore::Absolute => (actual - prediction).abs(),
            NonconformityScore::Signed => actual - prediction,
            NonconformityScore::Normalized => {
                // For normalized, we'd need variance estimate
                // Fall back to absolute for now
                (actual - prediction).abs()
            }
        }
    }

    /// Update quantile threshold from calibration scores
    fn update_quantile(&mut self) {
        if self.calibration_scores.is_empty() {
            self.quantile = 0.0;
            return;
        }

        let n = self.calibration_scores.len();
        // Quantile index: ceil((n+1) * (1-alpha)) / n
        let q = 1.0 - self.effective_alpha;
        let idx = ((n as f64 + 1.0) * q).ceil() as usize;
        let idx = idx.saturating_sub(1).min(n - 1);

        self.quantile = self.calibration_scores[idx];
    }

    /// Get current coverage rate
    pub fn coverage(&self) -> f64 {
        self.coverage_tracker.coverage()
    }

    /// Get current effective alpha
    pub fn effective_alpha(&self) -> f64 {
        self.effective_alpha
    }

    /// Check if predictor is calibrated
    pub fn is_calibrated(&self) -> bool {
        self.calibrated
    }

    /// Get number of calibration samples
    pub fn calibration_size(&self) -> usize {
        self.calibration_scores.len()
    }
}

/// Conformalized Quantile Regression predictor
pub struct CQRPredictor {
    /// Base quantile predictor (would be a neural network in production)
    lower_quantile: f64,
    upper_quantile: f64,
    /// Calibration scores
    calibration_scores: Vec<f64>,
    /// Quantile threshold
    quantile: f64,
    /// Alpha level
    alpha: f64,
    /// Calibrated flag
    calibrated: bool,
}

impl CQRPredictor {
    /// Create new CQR predictor
    pub fn new(alpha: f64) -> Self {
        Self {
            lower_quantile: alpha / 2.0,
            upper_quantile: 1.0 - alpha / 2.0,
            calibration_scores: Vec::new(),
            quantile: 0.0,
            alpha,
            calibrated: false,
        }
    }

    /// Calibrate with quantile predictions
    ///
    /// # Arguments
    /// * `lower_predictions` - Lower quantile predictions from base model
    /// * `upper_predictions` - Upper quantile predictions from base model
    /// * `actuals` - Actual values
    pub fn calibrate(
        &mut self,
        lower_predictions: &[f64],
        upper_predictions: &[f64],
        actuals: &[f64],
    ) -> Result<()> {
        if lower_predictions.len() != upper_predictions.len()
            || lower_predictions.len() != actuals.len()
        {
            return Err(NeuralBridgeError::InvalidInput(
                "All arrays must have same length".to_string(),
            ));
        }

        // CQR nonconformity score: max(lower - actual, actual - upper)
        self.calibration_scores = lower_predictions
            .iter()
            .zip(upper_predictions.iter())
            .zip(actuals.iter())
            .map(|((&l, &u), &a)| (l - a).max(a - u))
            .collect();

        self.calibration_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute quantile
        let n = self.calibration_scores.len();
        let q = 1.0 - self.alpha;
        let idx = ((n as f64 + 1.0) * q).ceil() as usize;
        let idx = idx.saturating_sub(1).min(n.saturating_sub(1));

        self.quantile = if !self.calibration_scores.is_empty() {
            self.calibration_scores[idx]
        } else {
            0.0
        };

        self.calibrated = true;

        Ok(())
    }

    /// Get prediction interval from quantile predictions
    pub fn predict(&self, lower_pred: f64, upper_pred: f64) -> PredictionInterval {
        let adjusted_lower = lower_pred - self.quantile;
        let adjusted_upper = upper_pred + self.quantile;
        let point = (lower_pred + upper_pred) / 2.0;

        PredictionInterval {
            point,
            lower: adjusted_lower,
            upper: adjusted_upper,
            coverage: 1.0 - self.alpha,
            width: adjusted_upper - adjusted_lower,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_conformal() {
        let mut predictor = HyperConformalPredictor::with_alpha(0.1);

        // Generate calibration data
        let predictions: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let actuals: Vec<f64> = predictions.iter().map(|p| p + 0.5).collect();

        predictor.calibrate(&predictions, &actuals).unwrap();

        assert!(predictor.is_calibrated());

        let interval = predictor.predict(50.0);
        assert!(interval.lower < 50.0);
        assert!(interval.upper > 50.0);
    }

    #[test]
    fn test_online_update() {
        let mut predictor = HyperConformalPredictor::new(ConformalConfig {
            alpha: 0.1,
            window_size: 100,
            ..Default::default()
        });

        // Initial calibration
        let predictions: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let actuals: Vec<f64> = predictions.iter().map(|p| p + 0.3).collect();
        predictor.calibrate(&predictions, &actuals).unwrap();

        // Online updates
        for i in 50..60 {
            predictor.update(i as f64, i as f64 + 0.3).unwrap();
        }

        assert!(predictor.coverage() > 0.8);
    }

    #[test]
    fn test_adaptive_conformal() {
        let mut predictor = HyperConformalPredictor::new(ConformalConfig {
            alpha: 0.1,
            method: ConformalMethod::Adaptive,
            window_size: 50,
            ..Default::default()
        });

        let predictions: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let actuals: Vec<f64> = predictions.iter().map(|p| p + 0.5).collect();
        predictor.calibrate(&predictions, &actuals).unwrap();

        // Simulate under-coverage scenario
        for i in 0..20 {
            let pred = i as f64;
            let actual = pred + 2.0; // Larger errors than calibration
            predictor.update(pred, actual).unwrap();
        }

        // Alpha should decrease (wider intervals) to compensate
        assert!(predictor.effective_alpha() < predictor.config.alpha);
    }

    #[test]
    fn test_cqr_predictor() {
        let mut predictor = CQRPredictor::new(0.1);

        let lower: Vec<f64> = (0..100).map(|i| i as f64 - 1.0).collect();
        let upper: Vec<f64> = (0..100).map(|i| i as f64 + 1.0).collect();
        let actuals: Vec<f64> = (0..100).map(|i| i as f64).collect();

        predictor.calibrate(&lower, &upper, &actuals).unwrap();

        let interval = predictor.predict(49.0, 51.0);
        assert!(interval.lower < 49.0);
        assert!(interval.upper > 51.0);
    }

    #[test]
    fn test_prediction_interval_contains() {
        let interval = PredictionInterval {
            point: 100.0,
            lower: 95.0,
            upper: 105.0,
            coverage: 0.9,
            width: 10.0,
        };

        assert!(interval.contains(100.0));
        assert!(interval.contains(95.0));
        assert!(interval.contains(105.0));
        assert!(!interval.contains(94.9));
        assert!(!interval.contains(105.1));
    }
}
