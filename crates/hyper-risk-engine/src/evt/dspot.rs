//! DSPOT (Drift-aware SPOT) for non-stationary streams.
//!
//! DSPOT extends SPOT to handle concept drift by:
//! 1. Maintaining a sliding window for threshold adaptation
//! 2. Detecting drift in the underlying distribution
//! 3. Re-calibrating when drift is detected
//!
//! ## Algorithm
//!
//! - Maintain exponentially weighted moving average of bulk distribution
//! - Detect drift using Page-Hinkley test or similar
//! - Adapt threshold when drift detected

use super::spot::{SpotConfig, SpotDetector, ExceedanceEvent};

/// DSPOT configuration.
#[derive(Debug, Clone)]
pub struct DspotConfig {
    /// Base SPOT configuration.
    pub spot_config: SpotConfig,
    /// Window size for drift detection.
    pub drift_window: usize,
    /// Drift detection threshold.
    pub drift_threshold: f64,
    /// Exponential weighting factor for EWMA.
    pub ewma_alpha: f64,
    /// Minimum observations between recalibrations.
    pub min_recalibration_interval: u64,
}

impl Default for DspotConfig {
    fn default() -> Self {
        Self {
            spot_config: SpotConfig::default(),
            drift_window: 500,
            drift_threshold: 2.0,
            ewma_alpha: 0.1,
            min_recalibration_interval: 1000,
        }
    }
}

/// Drift statistics.
#[derive(Debug, Clone)]
pub struct DriftStats {
    /// EWMA of recent values.
    pub ewma: f64,
    /// EWMA of squared values (for variance).
    pub ewma_sq: f64,
    /// Current variance estimate.
    pub variance: f64,
    /// Drift score (z-score of current mean vs historical).
    pub drift_score: f64,
    /// Is drift detected?
    pub drift_detected: bool,
}

/// DSPOT detector with drift adaptation.
#[derive(Debug)]
pub struct DspotDetector {
    /// Configuration.
    config: DspotConfig,
    /// Inner SPOT detector.
    spot: SpotDetector,
    /// Recent values for drift detection.
    recent_values: Vec<f64>,
    /// Historical mean estimate.
    historical_mean: f64,
    /// Historical variance estimate.
    historical_var: f64,
    /// EWMA of values.
    ewma: f64,
    /// EWMA of squared values.
    ewma_sq: f64,
    /// Last recalibration observation.
    last_recalibration: u64,
    /// Total drift events.
    drift_events: u64,
}

impl DspotDetector {
    /// Create new DSPOT detector.
    pub fn new(config: DspotConfig) -> Self {
        let spot = SpotDetector::new(config.spot_config.clone());
        Self {
            config,
            spot,
            recent_values: Vec::new(),
            historical_mean: 0.0,
            historical_var: 1.0,
            ewma: 0.0,
            ewma_sq: 0.0,
            last_recalibration: 0,
            drift_events: 0,
        }
    }

    /// Process new observation with drift detection.
    pub fn process(&mut self, value: f64) -> (Option<ExceedanceEvent>, DriftStats) {
        // Update EWMA
        let alpha = self.config.ewma_alpha;
        self.ewma = alpha * value + (1.0 - alpha) * self.ewma;
        self.ewma_sq = alpha * value * value + (1.0 - alpha) * self.ewma_sq;

        // Update recent values window
        self.recent_values.push(value);
        if self.recent_values.len() > self.config.drift_window {
            self.recent_values.remove(0);
        }

        // Detect drift
        let drift_stats = self.detect_drift();

        // If drift detected and enough time since last recalibration
        if drift_stats.drift_detected {
            let obs = self.spot.total_observations();
            if obs >= self.last_recalibration &&
               obs - self.last_recalibration >= self.config.min_recalibration_interval {
                self.recalibrate();
                self.last_recalibration = obs;
                self.drift_events += 1;
            }
        }

        // Process through SPOT
        let event = self.spot.process(value);

        (event, drift_stats)
    }

    /// Detect drift in the distribution.
    fn detect_drift(&self) -> DriftStats {
        let variance = self.ewma_sq - self.ewma * self.ewma;
        let variance = variance.max(1e-10); // Prevent division by zero

        // Calculate z-score of current EWMA vs historical mean
        let drift_score = if self.historical_var > 0.0 {
            (self.ewma - self.historical_mean) / self.historical_var.sqrt()
        } else {
            0.0
        };

        let drift_detected = drift_score.abs() > self.config.drift_threshold;

        DriftStats {
            ewma: self.ewma,
            ewma_sq: self.ewma_sq,
            variance,
            drift_score,
            drift_detected,
        }
    }

    /// Recalibrate the detector.
    fn recalibrate(&mut self) {
        // Update historical statistics
        if !self.recent_values.is_empty() {
            let mean: f64 = self.recent_values.iter().sum::<f64>() / self.recent_values.len() as f64;
            let var: f64 = self.recent_values.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / self.recent_values.len() as f64;

            self.historical_mean = mean;
            self.historical_var = var;
        }

        // Create new SPOT detector with fresh calibration
        // Note: In production, would need to handle this more carefully
        // to preserve some historical information
        let new_config = self.config.spot_config.clone();
        self.spot = SpotDetector::new(new_config);

        // Feed recent values for new calibration
        for &v in &self.recent_values {
            self.spot.process(v);
        }
    }

    /// Get underlying SPOT detector.
    pub fn spot(&self) -> &SpotDetector {
        &self.spot
    }

    /// Get total drift events.
    pub fn drift_events(&self) -> u64 {
        self.drift_events
    }

    /// Check if calibrated.
    pub fn is_calibrated(&self) -> bool {
        self.spot.is_calibrated()
    }

    /// Get current threshold.
    pub fn threshold(&self) -> f64 {
        self.spot.threshold()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dspot_creation() {
        let config = DspotConfig::default();
        let detector = DspotDetector::new(config);
        assert!(!detector.is_calibrated());
    }

    #[test]
    fn test_dspot_processing() {
        let config = DspotConfig {
            spot_config: SpotConfig {
                initial_batch_size: 50,
                ..Default::default()
            },
            drift_window: 20,
            ..Default::default()
        };
        let mut detector = DspotDetector::new(config);

        // Process normal values
        for i in 0..100 {
            let (_event, _stats) = detector.process(i as f64 / 10.0);
            // Should calibrate after initial batch
            if i >= 50 {
                assert!(detector.is_calibrated());
            }
        }
    }

    #[test]
    fn test_drift_detection() {
        let config = DspotConfig {
            spot_config: SpotConfig {
                initial_batch_size: 50,
                ..Default::default()
            },
            drift_window: 20,
            drift_threshold: 1.5,
            ewma_alpha: 0.2,
            min_recalibration_interval: 10,
        };
        let mut detector = DspotDetector::new(config);

        // Stable period
        for _ in 0..100 {
            detector.process(1.0);
        }

        // Drift period (sudden shift in mean)
        for _ in 0..50 {
            let (_, _stats) = detector.process(10.0);
            // Should detect drift after enough observations
        }

        // Should have detected drift
        assert!(detector.drift_events() > 0);
    }
}
