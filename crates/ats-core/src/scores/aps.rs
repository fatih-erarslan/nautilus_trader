//! Adaptive Prediction Sets (APS)
//!
//! Implementation of the APS nonconformity score from:
//!
//! Romano, Y., Sesia, M., & Candès, E. (2020).
//! "Classification with Valid and Adaptive Coverage." NeurIPS 2020.
//!
//! # Mathematical Definition
//!
//! APS is RAPS without regularization:
//!
//! ```text
//! s(x,y) = Σ_{j: π̂_j > π̂_y} π̂_j + u·π̂_y
//! ```
//!
//! Where:
//! - `π̂_j`: softmax probability for class j
//! - `u`: uniform random variable in [0,1] for tie-breaking
//!
//! # Relation to RAPS
//!
//! APS is equivalent to RAPS with λ=0 (no regularization).

use crate::NonconformityScorer;

#[derive(Clone, Debug)]
pub struct ApsConfig {
    /// Random tie-breaking for equal probabilities
    pub randomize_ties: bool,
}

impl Default for ApsConfig {
    fn default() -> Self {
        Self {
            randomize_ties: true,
        }
    }
}

/// APS nonconformity scorer
///
/// # Example
///
/// ```
/// use ats_core::scores::{ApsConfig, ApsScorer, NonconformityScorer};
///
/// let config = ApsConfig::default();
/// let scorer = ApsScorer::new(config);
///
/// let softmax = vec![0.6, 0.3, 0.1];
/// let true_label = 1;
/// let u = 0.5;
///
/// let score = scorer.score(&softmax, true_label, u);
/// assert!(score > 0.0);
/// ```
pub struct ApsScorer {
    config: ApsConfig,
}

impl ApsScorer {
    /// Create a new APS scorer with the given configuration
    pub fn new(config: ApsConfig) -> Self {
        Self { config }
    }

    /// Create APS scorer with default configuration
    pub fn default() -> Self {
        Self::new(ApsConfig::default())
    }

    /// Get current configuration
    pub fn config(&self) -> &ApsConfig {
        &self.config
    }
}

impl NonconformityScorer for ApsScorer {
    #[inline]
    fn score(&self, softmax_probs: &[f32], true_label: usize, u: f32) -> f32 {
        let n_classes = softmax_probs.len();

        debug_assert!(true_label < n_classes, "true_label out of bounds");
        debug_assert!((0.0..=1.0).contains(&u), "u must be in [0,1]");

        // Get sorted indices (descending by probability)
        let mut indices: Vec<usize> = (0..n_classes).collect();
        indices.sort_by(|&a, &b| {
            softmax_probs[b]
                .partial_cmp(&softmax_probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Find rank of true label
        let rank = indices
            .iter()
            .position(|&i| i == true_label)
            .expect("true_label not found in indices");

        // Cumulative sum up to (but not including) true label
        let cumsum: f32 = indices[..rank].iter().map(|&i| softmax_probs[i]).sum();

        // Add randomized portion of true label's probability
        let true_prob = softmax_probs[true_label];
        cumsum + u * true_prob
    }

    fn score_batch(
        &self,
        softmax_batch: &[Vec<f32>],
        labels: &[usize],
        u_values: &[f32],
    ) -> Vec<f32> {
        debug_assert_eq!(softmax_batch.len(), labels.len());
        debug_assert_eq!(softmax_batch.len(), u_values.len());

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
    fn test_aps_basic() {
        let config = ApsConfig::default();
        let scorer = ApsScorer::new(config);

        let softmax = vec![0.6, 0.3, 0.1];
        let true_label = 1;
        let u = 0.5;

        let score = scorer.score(&softmax, true_label, u);

        // cumsum = 0.6, base = 0.6 + 0.5 * 0.3 = 0.75
        assert!((score - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_aps_top_class() {
        let config = ApsConfig::default();
        let scorer = ApsScorer::new(config);

        let softmax = vec![0.7, 0.2, 0.1];
        let true_label = 0;
        let u = 0.5;

        let score = scorer.score(&softmax, true_label, u);

        // cumsum = 0, score = 0.5 * 0.7 = 0.35
        assert!((score - 0.35).abs() < 1e-6);
    }

    #[test]
    fn test_aps_vs_raps() {
        // APS should match RAPS with lambda=0
        use crate::scores::raps::{RapsConfig, RapsScorer};

        let aps_config = ApsConfig::default();
        let aps_scorer = ApsScorer::new(aps_config);

        let raps_config = RapsConfig {
            lambda: 0.0,
            k_reg: 0,
            randomize_ties: true,
        };
        let raps_scorer = RapsScorer::new(raps_config);

        let softmax = vec![0.6, 0.3, 0.1];
        let true_label = 1;
        let u = 0.5;

        let aps_score = aps_scorer.score(&softmax, true_label, u);
        let raps_score = raps_scorer.score(&softmax, true_label, u);

        assert!((aps_score - raps_score).abs() < 1e-6);
    }

    #[test]
    fn test_aps_batch() {
        let config = ApsConfig::default();
        let scorer = ApsScorer::new(config);

        let batch = vec![
            vec![0.6, 0.3, 0.1],
            vec![0.5, 0.3, 0.2],
            vec![0.4, 0.4, 0.2],
        ];
        let labels = vec![0, 1, 2];
        let u_values = vec![0.5, 0.5, 0.5];

        let scores = scorer.score_batch(&batch, &labels, &u_values);

        assert_eq!(scores.len(), 3);
        assert!(scores.iter().all(|&s| s >= 0.0 && s <= 1.0));
    }

    #[test]
    fn test_aps_boundary_u_values() {
        let config = ApsConfig::default();
        let scorer = ApsScorer::new(config);

        let softmax = vec![0.6, 0.3, 0.1];
        let true_label = 1;

        let score_u0 = scorer.score(&softmax, true_label, 0.0);
        let score_u1 = scorer.score(&softmax, true_label, 1.0);

        // cumsum = 0.6
        // u=0: 0.6 + 0.0 * 0.3 = 0.6
        // u=1: 0.6 + 1.0 * 0.3 = 0.9
        assert!((score_u0 - 0.6).abs() < 1e-6);
        assert!((score_u1 - 0.9).abs() < 1e-6);
        assert!(score_u1 > score_u0);
    }
}
