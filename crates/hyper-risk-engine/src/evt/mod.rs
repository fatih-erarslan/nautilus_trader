//! Streaming Extreme Value Theory (EVT) for tail risk monitoring.
//!
//! Implements SPOT (Streaming Peaks Over Threshold) and DSPOT
//! for real-time anomaly detection without storing full history.
//!
//! ## Scientific References
//! - Siffer et al. (2017): "Anomaly Detection in Streams with Extreme Value Theory"
//! - McNeil & Frey (2000): "Estimation of Tail-Related Risk Measures for Heteroscedastic Financial Time Series"
//! - Pickands (1975): "Statistical Inference Using Extreme Order Statistics"
//!
//! ## Algorithm Overview
//!
//! SPOT maintains a streaming estimate of GPD parameters for the tail distribution:
//! - γ (shape): Controls tail heaviness
//! - σ (scale): Controls spread
//! - Threshold t: Separates bulk from tail
//!
//! For each new observation x:
//! 1. If x > t: Update GPD parameters, check if anomaly
//! 2. If x ≤ t: Observation is in bulk distribution, continue

pub mod gpd;
pub mod spot;
pub mod dspot;

pub use gpd::{GPDParams, GPDEstimator, TailRiskEstimate};
pub use spot::{SpotConfig, SpotDetector, ExceedanceEvent};
pub use dspot::{DspotConfig, DspotDetector};

/// Main streaming EVT interface.
#[derive(Debug)]
pub struct StreamingEVT {
    /// SPOT detector for upper tail.
    spot_upper: SpotDetector,
    /// SPOT detector for lower tail (for drawdowns).
    spot_lower: SpotDetector,
    /// Total observations processed.
    total_observations: u64,
}

impl StreamingEVT {
    /// Create new streaming EVT processor.
    pub fn new(config: SpotConfig) -> Self {
        Self {
            spot_upper: SpotDetector::new(config.clone()),
            spot_lower: SpotDetector::new(config),
            total_observations: 0,
        }
    }

    /// Process new observation and check for tail events.
    ///
    /// # Arguments
    /// * `value` - New observation (e.g., return, loss)
    ///
    /// # Returns
    /// Tuple of (upper_exceedance, lower_exceedance)
    pub fn process(&mut self, value: f64) -> (Option<ExceedanceEvent>, Option<ExceedanceEvent>) {
        self.total_observations += 1;

        // Check upper tail (positive extremes)
        let upper = self.spot_upper.process(value);

        // Check lower tail (negative extremes - invert sign)
        let lower = self.spot_lower.process(-value).map(|mut e| {
            e.value = -e.value; // Restore original sign
            e.threshold = -e.threshold;
            e
        });

        (upper, lower)
    }

    /// Get upper tail risk estimate.
    pub fn upper_tail_risk(&self, confidence: f64) -> Option<TailRiskEstimate> {
        self.spot_upper.tail_risk_estimate(confidence)
    }

    /// Get lower tail risk estimate.
    pub fn lower_tail_risk(&self, confidence: f64) -> Option<TailRiskEstimate> {
        self.spot_lower.tail_risk_estimate(confidence).map(|mut e| {
            e.var = -e.var;
            e.es = -e.es;
            e
        })
    }

    /// Get VaR estimate at given confidence level.
    pub fn var(&self, confidence: f64) -> Option<f64> {
        self.upper_tail_risk(confidence).map(|t| t.var)
    }

    /// Get Expected Shortfall (CVaR) at given confidence level.
    pub fn es(&self, confidence: f64) -> Option<f64> {
        self.upper_tail_risk(confidence).map(|t| t.es)
    }

    /// Get total observations processed.
    pub fn total_observations(&self) -> u64 {
        self.total_observations
    }

    /// Check if EVT model is calibrated (has enough data).
    pub fn is_calibrated(&self) -> bool {
        self.spot_upper.is_calibrated() && self.spot_lower.is_calibrated()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_evt_creation() {
        let config = SpotConfig::default();
        let evt = StreamingEVT::new(config);
        assert_eq!(evt.total_observations(), 0);
        assert!(!evt.is_calibrated());
    }

    #[test]
    fn test_evt_processing() {
        let config = SpotConfig {
            initial_batch_size: 10,
            ..Default::default()
        };
        let mut evt = StreamingEVT::new(config);

        // Process some normal observations
        for i in 0..20 {
            let value = (i as f64 - 10.0) / 10.0;
            let (upper, _lower) = evt.process(value);
            // Early observations shouldn't trigger (model not calibrated)
            if i < 10 {
                assert!(upper.is_none());
            }
        }

        assert_eq!(evt.total_observations(), 20);
    }
}
