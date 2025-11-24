//! ATS-Core Integration Module
//!
//! Bridges ats-core's conformal prediction and temperature scaling with hyperphysics-neural.
//!
//! ## Features
//!
//! - **Conformal Prediction**: Distribution-free uncertainty quantification (<20μs)
//! - **Temperature Scaling**: Neural network calibration (<5μs)
//! - **Calibrated Backends**: Uncertainty-aware reasoning backends
//! - **27+ Architectures**: Full neural architecture catalog
//!
//! ## HFT Application
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │                  ATS Integration Layer                        │
//! │                                                               │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
//! │  │ Conformal   │  │ Temperature │  │ CalibratedFannBackend│  │
//! │  │ Predictor   │  │  Scaler     │  │ (uncertainty-aware)  │  │
//! │  │   <20μs     │  │    <5μs     │  │                      │  │
//! │  └──────┬──────┘  └──────┬──────┘  └──────────┬───────────┘  │
//! │         │                │                     │              │
//! │         └────────────────┼─────────────────────┘              │
//! │                          ▼                                    │
//! │              ┌───────────────────────┐                        │
//! │              │    FannNetwork        │                        │
//! │              │    (ruv-FANN)         │                        │
//! │              └───────────────────────┘                        │
//! └──────────────────────────────────────────────────────────────┘
//! ```

pub mod conformal;
pub mod calibration;
pub mod backend;

#[cfg(feature = "architectures")]
pub mod architectures;

// Re-export key types from ats-core
pub use ats_core::{
    // Core types
    AtsCpConfig,
    AtsCpResult,
    AtsCpVariant,
    // Conformal prediction
    PredictionInterval,
    PredictionIntervals,
    // Temperature scaling
    TemperatureScaler,
    // Error handling
    AtsCoreError,
};

// Re-export from specific modules
pub use ats_core::conformal::ConformalPredictor as AtsConformalPredictor;
pub use ats_core::types::TemperatureScalingResult;

// Re-export our bridge types
pub use conformal::{ConformalConfig, FastConformalPredictor, UncertaintyBounds};
pub use calibration::{CalibrationConfig, CalibratedPrediction, NeuralCalibrator};
pub use backend::{CalibratedFannBackend, CalibratedFannConfig, UncertaintyAwareResult};

use crate::error::NeuralResult;

/// ATS-CP variants optimized for different use cases
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConformalVariant {
    /// Gamma Quantile - fastest, best for HFT
    GQ,
    /// Adaptive Quantile - adapts to data distribution
    AQ,
    /// Modified Gamma Quantile - better coverage
    MGQ,
    /// Modified Adaptive Quantile - most robust
    MAQ,
}

impl Default for ConformalVariant {
    fn default() -> Self {
        Self::GQ // Fastest for HFT
    }
}

impl From<ConformalVariant> for AtsCpVariant {
    fn from(v: ConformalVariant) -> Self {
        match v {
            ConformalVariant::GQ => AtsCpVariant::GQ,
            ConformalVariant::AQ => AtsCpVariant::AQ,
            ConformalVariant::MGQ => AtsCpVariant::MGQ,
            ConformalVariant::MAQ => AtsCpVariant::MAQ,
        }
    }
}

/// Quick uncertainty quantification for a prediction
pub fn quick_uncertainty(
    prediction: f64,
    calibration_scores: &[f64],
    confidence: f64,
) -> NeuralResult<(f64, f64)> {
    if calibration_scores.is_empty() {
        return Err(crate::error::NeuralError::InvalidInput(
            "calibration_scores cannot be empty".into(),
        ));
    }

    // Fast quantile computation
    let mut sorted = calibration_scores.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let idx = ((confidence * sorted.len() as f64) as usize).min(sorted.len() - 1);
    let quantile = sorted[idx];

    Ok((prediction - quantile, prediction + quantile))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conformal_variant_default() {
        let variant = ConformalVariant::default();
        assert_eq!(variant, ConformalVariant::GQ);
    }

    #[test]
    fn test_quick_uncertainty() {
        let prediction = 100.0;
        let calibration = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let (lower, upper) = quick_uncertainty(prediction, &calibration, 0.95).unwrap();

        assert!(lower < prediction);
        assert!(upper > prediction);
        assert!((upper - lower) > 0.0);
    }

    #[test]
    fn test_variant_conversion() {
        let gq: AtsCpVariant = ConformalVariant::GQ.into();
        assert!(matches!(gq, AtsCpVariant::GQ));

        let maq: AtsCpVariant = ConformalVariant::MAQ.into();
        assert!(matches!(maq, AtsCpVariant::MAQ));
    }
}
