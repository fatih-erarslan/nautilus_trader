//! Least Ambiguous Classifiers (LAC)
//!
//! Implementation based on:
//!
//! Stutz, D., Dvijotham, K. D., Cemgil, A. T., & Doucet, A. (2022).
//! "Learning Optimal Conformal Classifiers." ICLR 2022.
//!
//! # Mathematical Definition
//!
//! LAC uses learned weights to optimize prediction set size:
//!
//! ```text
//! s(x,y) = Σ_{j ≠ y} w_j · π̂_j
//! ```
//!
//! Where w_j are learned weights that minimize expected set size
//! while maintaining coverage guarantees.

use crate::NonconformityScorer;

#[derive(Clone, Debug)]
pub struct LacConfig {
    /// Learned class weights (if None, uses uniform weights)
    pub class_weights: Option<Vec<f32>>,

    /// Random tie-breaking
    pub randomize_ties: bool,
}

impl Default for LacConfig {
    fn default() -> Self {
        Self {
            class_weights: None,
            randomize_ties: true,
        }
    }
}

/// LAC nonconformity scorer
///
/// # Example
///
/// ```
/// use ats_core::scores::{LacConfig, LacScorer, NonconformityScorer};
///
/// let config = LacConfig::default();
/// let scorer = LacScorer::new(config);
///
/// let softmax = vec![0.6, 0.3, 0.1];
/// let true_label = 1;
/// let u = 0.5;
///
/// let score = scorer.score(&softmax, true_label, u);
/// assert!(score > 0.0);
/// ```
pub struct LacScorer {
    config: LacConfig,
}

impl LacScorer {
    /// Create a new LAC scorer with the given configuration
    pub fn new(config: LacConfig) -> Self {
        Self { config }
    }

    /// Create LAC scorer with default configuration (uniform weights)
    pub fn default() -> Self {
        Self::new(LacConfig::default())
    }

    /// Create LAC scorer with learned weights
    pub fn with_weights(class_weights: Vec<f32>) -> Self {
        Self::new(LacConfig {
            class_weights: Some(class_weights),
            randomize_ties: true,
        })
    }

    /// Get current configuration
    pub fn config(&self) -> &LacConfig {
        &self.config
    }

    /// Get uniform weights for n classes
    fn get_weights(&self, n_classes: usize) -> Vec<f32> {
        match &self.config.class_weights {
            Some(weights) => {
                debug_assert_eq!(weights.len(), n_classes, "Weight dimension mismatch");
                weights.clone()
            }
            None => vec![1.0; n_classes], // Uniform weights
        }
    }
}

impl NonconformityScorer for LacScorer {
    #[inline]
    fn score(&self, softmax_probs: &[f32], true_label: usize, u: f32) -> f32 {
        let n_classes = softmax_probs.len();

        debug_assert!(true_label < n_classes, "true_label out of bounds");
        debug_assert!((0.0..=1.0).contains(&u), "u must be in [0,1]");

        let weights = self.get_weights(n_classes);

        // Compute weighted sum of all classes except true label
        let mut weighted_sum: f32 = (0..n_classes)
            .filter(|&j| j != true_label)
            .map(|j| weights[j] * softmax_probs[j])
            .sum();

        // Add randomized portion of true label contribution
        // This maintains the exchangeability property
        weighted_sum += u * weights[true_label] * softmax_probs[true_label];

        weighted_sum
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
    fn test_lac_uniform_weights() {
        let config = LacConfig::default();
        let scorer = LacScorer::new(config);

        let softmax = vec![0.6, 0.3, 0.1];
        let true_label = 1;
        let u = 0.5;

        let score = scorer.score(&softmax, true_label, u);

        // Weighted sum of all except true label: 1.0 * 0.6 + 1.0 * 0.1 = 0.7
        // Plus u * weight * true_prob: 0.5 * 1.0 * 0.3 = 0.15
        // Total: 0.7 + 0.15 = 0.85
        assert!((score - 0.85).abs() < 1e-6);
    }

    #[test]
    fn test_lac_custom_weights() {
        let weights = vec![2.0, 1.0, 0.5];
        let scorer = LacScorer::with_weights(weights.clone());

        let softmax = vec![0.6, 0.3, 0.1];
        let true_label = 1;
        let u = 0.5;

        let score = scorer.score(&softmax, true_label, u);

        // Weighted sum: 2.0 * 0.6 + 0.5 * 0.1 = 1.2 + 0.05 = 1.25
        // Plus: 0.5 * 1.0 * 0.3 = 0.15
        // Total: 1.25 + 0.15 = 1.40
        assert!((score - 1.40).abs() < 1e-6);
    }

    #[test]
    fn test_lac_batch() {
        let config = LacConfig::default();
        let scorer = LacScorer::new(config);

        let batch = vec![
            vec![0.6, 0.3, 0.1],
            vec![0.5, 0.3, 0.2],
            vec![0.4, 0.4, 0.2],
        ];
        let labels = vec![0, 1, 2];
        let u_values = vec![0.5, 0.5, 0.5];

        let scores = scorer.score_batch(&batch, &labels, &u_values);

        assert_eq!(scores.len(), 3);
        assert!(scores.iter().all(|&s| s >= 0.0));
    }

    #[test]
    fn test_lac_deterministic_with_u_05() {
        let config = LacConfig {
            randomize_ties: false,
            ..Default::default()
        };
        let scorer = LacScorer::new(config);

        let softmax = vec![0.6, 0.3, 0.1];
        let true_label = 1;

        let score1 = scorer.score(&softmax, true_label, 0.5);
        let score2 = scorer.score(&softmax, true_label, 0.5);

        assert_eq!(score1, score2);
    }

    #[test]
    fn test_lac_u_boundary_values() {
        let scorer = LacScorer::default();

        let softmax = vec![0.6, 0.3, 0.1];
        let true_label = 1;

        let score_u0 = scorer.score(&softmax, true_label, 0.0);
        let score_u1 = scorer.score(&softmax, true_label, 1.0);

        // u=0: sum of others = 0.7
        // u=1: sum of others + true_prob = 0.7 + 0.3 = 1.0
        assert!((score_u0 - 0.7).abs() < 1e-6);
        assert!((score_u1 - 1.0).abs() < 1e-6);
        assert!(score_u1 > score_u0);
    }

    #[test]
    fn test_lac_weight_effect() {
        // Higher weight on class should increase its contribution
        let weights1 = vec![1.0, 1.0, 1.0];
        let weights2 = vec![10.0, 1.0, 1.0]; // Much higher weight on class 0

        let scorer1 = LacScorer::with_weights(weights1);
        let scorer2 = LacScorer::with_weights(weights2);

        let softmax = vec![0.6, 0.3, 0.1];
        let true_label = 1; // Class 0 is not the true label

        let score1 = scorer1.score(&softmax, true_label, 0.5);
        let score2 = scorer2.score(&softmax, true_label, 0.5);

        assert!(
            score2 > score1,
            "Higher weight on non-true class should increase score"
        );
    }
}
