//! Nonconformity Scores for Adaptive Conformal Prediction
//!
//! This module implements state-of-the-art nonconformity scoring functions
//! based on peer-reviewed research:
//!
//! - **RAPS**: Regularized Adaptive Prediction Sets (Romano et al., 2020)
//! - **APS**: Adaptive Prediction Sets (Romano et al., 2020)
//! - **SAPS**: Sorted Adaptive Prediction Sets
//! - **THR**: Threshold-based scores
//! - **LAC**: Least Ambiguous Classifiers (Stutz et al., 2022)
//!
//! # References
//!
//! - Romano, Y., Sesia, M., & Candès, E. (2020). "Classification with Valid
//!   and Adaptive Coverage." NeurIPS 2020.
//! - Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., & Schuster, T. (2021).
//!   "Uncertainty Sets for Image Classifiers using Conformal Prediction." ICLR 2021.
//! - Stutz, D., Dvijotham, K. D., Cemgil, A. T., & Doucet, A. (2022).
//!   "Learning Optimal Conformal Classifiers." ICLR 2022.

pub mod raps;
pub mod aps;
pub mod saps;
pub mod thr;
pub mod lac;

pub use raps::{RapsConfig, RapsScorer};
pub use aps::{ApsConfig, ApsScorer};
pub use saps::{SapsConfig, SapsScorer};
pub use thr::ThresholdScorer;
pub use lac::{LacConfig, LacScorer};

/// Common trait for all nonconformity scorers
pub trait NonconformityScorer: Send + Sync {
    /// Compute nonconformity score for a single sample
    ///
    /// # Arguments
    /// * `softmax_probs` - Softmax probabilities for all classes
    /// * `true_label` - Index of the true class
    /// * `u` - Random uniform [0,1] for tie-breaking
    ///
    /// # Returns
    /// Nonconformity score s(x,y)
    fn score(&self, softmax_probs: &[f32], true_label: usize, u: f32) -> f32;

    /// Vectorized batch scoring
    fn score_batch(
        &self,
        softmax_batch: &[Vec<f32>],
        labels: &[usize],
        u_values: &[f32],
    ) -> Vec<f32> {
        softmax_batch
            .iter()
            .zip(labels.iter())
            .zip(u_values.iter())
            .map(|((probs, &label), &u)| self.score(probs, label, u))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_scorers_implemented() {
        // Smoke test to ensure all modules are properly exported
        let _raps_config = RapsConfig::default();
        let _aps_config = ApsConfig::default();
        let _saps_config = SapsConfig::default();
        let _lac_config = LacConfig::default();
    }
}
