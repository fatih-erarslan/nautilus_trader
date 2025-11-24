//! Temperature Scaling and Calibration Bridge
//!
//! Wraps ats-core's temperature scaling for neural network calibration.

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::error::{NeuralError, NeuralResult};

/// Configuration for neural network calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// Enable temperature scaling
    pub temperature_scaling: bool,
    /// Enable Platt scaling
    pub platt_scaling: bool,
    /// Enable isotonic regression
    pub isotonic_regression: bool,
    /// Initial temperature value
    pub initial_temperature: f64,
    /// Minimum temperature bound
    pub min_temperature: f64,
    /// Maximum temperature bound
    pub max_temperature: f64,
    /// Target latency in microseconds
    pub target_latency_us: u64,
    /// Optimization tolerance
    pub tolerance: f64,
    /// Maximum optimization iterations
    pub max_iterations: usize,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            temperature_scaling: true,
            platt_scaling: false,
            isotonic_regression: false,
            initial_temperature: 1.0,
            min_temperature: 0.01,
            max_temperature: 10.0,
            target_latency_us: 5, // 5μs for HFT
            tolerance: 1e-6,
            max_iterations: 100,
        }
    }
}

impl CalibrationConfig {
    /// HFT-optimized configuration
    pub fn hft() -> Self {
        Self {
            temperature_scaling: true,
            platt_scaling: false,
            isotonic_regression: false,
            initial_temperature: 1.0,
            min_temperature: 0.1,
            max_temperature: 5.0,
            target_latency_us: 3,
            tolerance: 1e-4,
            max_iterations: 50,
        }
    }

    /// Full calibration configuration
    pub fn full() -> Self {
        Self {
            temperature_scaling: true,
            platt_scaling: true,
            isotonic_regression: true,
            initial_temperature: 1.0,
            min_temperature: 0.01,
            max_temperature: 10.0,
            target_latency_us: 100,
            tolerance: 1e-8,
            max_iterations: 500,
        }
    }
}

/// Result of calibrated prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibratedPrediction {
    /// Original raw prediction
    pub raw_prediction: f64,
    /// Calibrated prediction after scaling
    pub calibrated_prediction: f64,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Applied temperature
    pub temperature: f64,
    /// ATS calibration score
    pub ats_score: f64,
    /// Computation time in nanoseconds
    pub latency_ns: u64,
}

/// Neural network calibrator
pub struct NeuralCalibrator {
    /// Configuration
    config: CalibrationConfig,
    /// Current optimal temperature
    optimal_temperature: f64,
    /// Exponential lookup table for fast computation
    exp_cache: Vec<f64>,
    /// Cache scale factor
    cache_scale: f64,
    /// Statistics
    total_calibrations: u64,
    total_latency_ns: u64,
}

impl NeuralCalibrator {
    /// Create new calibrator
    pub fn new(config: CalibrationConfig) -> Self {
        // Pre-compute exponential lookup table
        let cache_size = 10000;
        let cache_scale = 100.0;
        let mut exp_cache = Vec::with_capacity(cache_size);

        for i in 0..cache_size {
            let x = (i as f64 - cache_size as f64 / 2.0) / cache_scale;
            exp_cache.push(x.exp());
        }

        Self {
            optimal_temperature: config.initial_temperature,
            config,
            exp_cache,
            cache_scale,
            total_calibrations: 0,
            total_latency_ns: 0,
        }
    }

    /// Create HFT-optimized calibrator
    pub fn hft() -> Self {
        Self::new(CalibrationConfig::hft())
    }

    /// Apply temperature scaling to logits
    pub fn scale(&self, logits: &[f64], temperature: f64) -> NeuralResult<Vec<f64>> {
        if temperature <= 0.0 {
            return Err(NeuralError::InvalidInput(
                "Temperature must be positive".into(),
            ));
        }

        let inv_temp = 1.0 / temperature;
        Ok(logits.iter().map(|&x| self.fast_exp(x * inv_temp)).collect())
    }

    /// Apply softmax with temperature scaling
    pub fn softmax_with_temperature(&self, logits: &[f64], temperature: f64) -> NeuralResult<Vec<f64>> {
        if logits.is_empty() {
            return Err(NeuralError::InvalidInput("Logits cannot be empty".into()));
        }

        if temperature <= 0.0 {
            return Err(NeuralError::InvalidInput(
                "Temperature must be positive".into(),
            ));
        }

        let inv_temp = 1.0 / temperature;

        // Find max for numerical stability
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Compute scaled exponentials
        let exp_values: Vec<f64> = logits
            .iter()
            .map(|&x| self.fast_exp((x - max_logit) * inv_temp))
            .collect();

        let sum: f64 = exp_values.iter().sum();

        if sum <= 0.0 {
            return Err(NeuralError::ComputeError(
                "Softmax sum is zero".into(),
            ));
        }

        Ok(exp_values.iter().map(|&x| x / sum).collect())
    }

    /// Calibrate a single prediction
    pub fn calibrate(&mut self, prediction: f64) -> NeuralResult<CalibratedPrediction> {
        let start = Instant::now();

        let scaled = self.fast_exp(prediction / self.optimal_temperature);
        let confidence = self.compute_confidence(scaled);

        let latency_ns = start.elapsed().as_nanos() as u64;
        self.total_calibrations += 1;
        self.total_latency_ns += latency_ns;

        Ok(CalibratedPrediction {
            raw_prediction: prediction,
            calibrated_prediction: scaled,
            confidence,
            temperature: self.optimal_temperature,
            ats_score: confidence, // Simplified ATS score
            latency_ns,
        })
    }

    /// Calibrate batch of predictions
    pub fn calibrate_batch(&mut self, predictions: &[f64]) -> NeuralResult<Vec<CalibratedPrediction>> {
        let start = Instant::now();

        let results: Vec<_> = predictions
            .iter()
            .map(|&pred| {
                let scaled = self.fast_exp(pred / self.optimal_temperature);
                let confidence = self.compute_confidence(scaled);
                CalibratedPrediction {
                    raw_prediction: pred,
                    calibrated_prediction: scaled,
                    confidence,
                    temperature: self.optimal_temperature,
                    ats_score: confidence,
                    latency_ns: 0,
                }
            })
            .collect();

        let total_latency = start.elapsed().as_nanos() as u64;
        let per_pred_latency = total_latency / predictions.len().max(1) as u64;

        Ok(results
            .into_iter()
            .map(|mut c| {
                c.latency_ns = per_pred_latency;
                c
            })
            .collect())
    }

    /// Optimize temperature using calibration data
    pub fn optimize_temperature(
        &mut self,
        predictions: &[f64],
        targets: &[f64],
    ) -> NeuralResult<f64> {
        if predictions.len() != targets.len() {
            return Err(NeuralError::DimensionMismatch {
                input_dim: predictions.len(),
                expected_dim: targets.len(),
            });
        }

        // Binary search for optimal temperature
        let mut low = self.config.min_temperature;
        let mut high = self.config.max_temperature;
        let mut best_temp = self.config.initial_temperature;

        for _ in 0..self.config.max_iterations {
            if (high - low) < self.config.tolerance {
                break;
            }

            let mid = (low + high) / 2.0;
            let error = self.compute_calibration_error(predictions, targets, mid)?;

            if error > 0.0 {
                high = mid;
            } else {
                low = mid;
            }

            best_temp = mid;
        }

        self.optimal_temperature = best_temp;
        Ok(best_temp)
    }

    /// Compute calibration error (Expected Calibration Error)
    fn compute_calibration_error(
        &self,
        predictions: &[f64],
        targets: &[f64],
        temperature: f64,
    ) -> NeuralResult<f64> {
        let inv_temp = 1.0 / temperature;
        let mut total_error = 0.0;

        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            let scaled = self.fast_exp(pred * inv_temp);
            let diff = scaled - target;
            total_error += diff * diff;
        }

        Ok(total_error / predictions.len() as f64)
    }

    /// Compute confidence from scaled prediction
    fn compute_confidence(&self, scaled: f64) -> f64 {
        // Sigmoid-based confidence mapping
        1.0 / (1.0 + (-scaled).exp())
    }

    /// Fast exponential using lookup table
    fn fast_exp(&self, x: f64) -> f64 {
        let clamped = x.clamp(-50.0, 50.0);
        let idx = (clamped * self.cache_scale + self.exp_cache.len() as f64 / 2.0) as usize;

        if idx < self.exp_cache.len() {
            self.exp_cache[idx]
        } else {
            clamped.exp()
        }
    }

    /// Get current optimal temperature
    pub fn temperature(&self) -> f64 {
        self.optimal_temperature
    }

    /// Set temperature manually
    pub fn set_temperature(&mut self, temp: f64) {
        self.optimal_temperature = temp.clamp(self.config.min_temperature, self.config.max_temperature);
    }

    /// Get average latency
    pub fn avg_latency_ns(&self) -> f64 {
        if self.total_calibrations == 0 {
            0.0
        } else {
            self.total_latency_ns as f64 / self.total_calibrations as f64
        }
    }

    /// Get configuration
    pub fn config(&self) -> &CalibrationConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibrator_creation() {
        let calibrator = NeuralCalibrator::hft();
        assert_eq!(calibrator.temperature(), 1.0);
    }

    #[test]
    fn test_temperature_scaling() {
        let calibrator = NeuralCalibrator::new(CalibrationConfig::default());
        let logits = vec![1.0, 2.0, 3.0];

        let scaled = calibrator.scale(&logits, 2.0).unwrap();
        assert_eq!(scaled.len(), 3);

        // Lower temperature should give more extreme values
        let scaled_low = calibrator.scale(&logits, 0.5).unwrap();
        assert!(scaled_low[2] > scaled[2]);
    }

    #[test]
    fn test_softmax_with_temperature() {
        let calibrator = NeuralCalibrator::new(CalibrationConfig::default());
        let logits = vec![1.0, 2.0, 3.0];

        let probs = calibrator.softmax_with_temperature(&logits, 1.0).unwrap();

        // Should sum to 1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Should be ordered
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
    }

    #[test]
    fn test_calibrate_prediction() {
        let mut calibrator = NeuralCalibrator::new(CalibrationConfig::default());

        let result = calibrator.calibrate(1.0).unwrap();

        assert_eq!(result.raw_prediction, 1.0);
        assert!(result.calibrated_prediction > 0.0);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    #[test]
    fn test_batch_calibration() {
        let mut calibrator = NeuralCalibrator::new(CalibrationConfig::default());
        let predictions = vec![0.5, 1.0, 1.5, 2.0];

        let results = calibrator.calibrate_batch(&predictions).unwrap();

        assert_eq!(results.len(), 4);
        for r in &results {
            assert!(r.confidence >= 0.0 && r.confidence <= 1.0);
        }
    }

    #[test]
    fn test_temperature_optimization() {
        let mut calibrator = NeuralCalibrator::new(CalibrationConfig::default());

        let predictions = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let targets = vec![0.9, 1.8, 2.7, 3.6, 4.5];

        let optimal = calibrator.optimize_temperature(&predictions, &targets).unwrap();

        assert!(optimal > 0.0);
        assert!(optimal <= calibrator.config().max_temperature);
    }

    #[test]
    fn test_set_temperature() {
        let mut calibrator = NeuralCalibrator::new(CalibrationConfig::default());

        calibrator.set_temperature(2.5);
        assert_eq!(calibrator.temperature(), 2.5);

        // Should clamp to bounds
        calibrator.set_temperature(100.0);
        assert_eq!(calibrator.temperature(), calibrator.config().max_temperature);
    }
}
