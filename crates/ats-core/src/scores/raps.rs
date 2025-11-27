//! Regularized Adaptive Prediction Sets (RAPS)
//!
//! Implementation of the RAPS nonconformity score from:
//!
//! Romano, Y., Sesia, M., & Candès, E. (2020).
//! "Classification with Valid and Adaptive Coverage." NeurIPS 2020.
//!
//! # Mathematical Definition
//!
//! The RAPS score is defined as:
//!
//! ```text
//! s(x,y) = Σ_{j: π̂_j > π̂_y} π̂_j + u·π̂_y + λ·(o(x,y) - k_reg)^+
//! ```
//!
//! Where:
//! - `π̂_j`: softmax probability for class j
//! - `o(x,y)`: rank of true class y (1-indexed, sorted by descending probability)
//! - `k_reg`: regularization target rank
//! - `λ`: regularization strength
//! - `u`: uniform random variable in [0,1] for tie-breaking
//! - `(z)^+`: max(0, z)
//!
//! # Performance Target
//!
//! RAPS scoring must achieve <3μs per sample on production hardware.

use crate::NonconformityScorer;

#[derive(Clone, Debug)]
pub struct RapsConfig {
    /// Regularization strength λ
    ///
    /// Controls the penalty for large prediction sets. Higher values produce
    /// smaller sets but may sacrifice coverage. Typical range: [0.001, 0.1].
    pub lambda: f32,

    /// Target rank for regularization k_reg
    ///
    /// The rank threshold above which regularization is applied.
    /// Classes ranked lower than k_reg incur a penalty. Typical range: [3, 10].
    pub k_reg: usize,

    /// Random tie-breaking for equal probabilities
    ///
    /// When true, uses uniform random u for tie-breaking. When false, uses u=0.5.
    pub randomize_ties: bool,
}

impl Default for RapsConfig {
    fn default() -> Self {
        Self {
            lambda: 0.01,
            k_reg: 5,
            randomize_ties: true,
        }
    }
}

/// RAPS nonconformity scorer
///
/// # Example
///
/// ```
/// use ats_core::scores::{RapsConfig, RapsScorer, NonconformityScorer};
///
/// let config = RapsConfig::default();
/// let scorer = RapsScorer::new(config);
///
/// let softmax = vec![0.6, 0.3, 0.1];
/// let true_label = 1;
/// let u = 0.5;
///
/// let score = scorer.score(&softmax, true_label, u);
/// assert!(score > 0.0);
/// ```
pub struct RapsScorer {
    config: RapsConfig,
}

impl RapsScorer {
    /// Create a new RAPS scorer with the given configuration
    pub fn new(config: RapsConfig) -> Self {
        Self { config }
    }

    /// Create RAPS scorer with default configuration
    pub fn default() -> Self {
        Self::new(RapsConfig::default())
    }

    /// Get current configuration
    pub fn config(&self) -> &RapsConfig {
        &self.config
    }
}

impl NonconformityScorer for RapsScorer {
    #[inline]
    fn score(&self, softmax_probs: &[f32], true_label: usize, u: f32) -> f32 {
        let n_classes = softmax_probs.len();

        // Validate inputs
        debug_assert!(true_label < n_classes, "true_label out of bounds");
        debug_assert!((0.0..=1.0).contains(&u), "u must be in [0,1]");

        // Get sorted indices (descending by probability)
        let mut indices: Vec<usize> = (0..n_classes).collect();
        indices.sort_by(|&a, &b| {
            softmax_probs[b]
                .partial_cmp(&softmax_probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Find rank of true label (0-indexed in array, but conceptually 1-indexed)
        let rank = indices
            .iter()
            .position(|&i| i == true_label)
            .expect("true_label not found in indices");

        // Cumulative sum up to (but not including) true label
        // This represents Σ_{j: π̂_j > π̂_y} π̂_j
        let cumsum: f32 = indices[..rank].iter().map(|&i| softmax_probs[i]).sum();

        // Add randomized portion of true label's probability
        let true_prob = softmax_probs[true_label];
        let base_score = cumsum + u * true_prob;

        // Regularization term: λ·(rank - k_reg)^+
        // Note: rank is 0-indexed, but conceptually we treat it as 1-indexed rank
        let effective_rank = rank + 1; // Convert to 1-indexed rank
        let reg_term = if effective_rank > self.config.k_reg {
            self.config.lambda * (effective_rank - self.config.k_reg) as f32
        } else {
            0.0
        };

        base_score + reg_term
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
    fn test_raps_basic() {
        let config = RapsConfig::default();
        let scorer = RapsScorer::new(config);

        // Simple 3-class example
        let softmax = vec![0.6, 0.3, 0.1];
        let true_label = 1; // Second class
        let u = 0.5;

        let score = scorer.score(&softmax, true_label, u);

        // Expected: cumsum of probs > 0.3 is 0.6
        // Plus u * 0.3 = 0.5 * 0.3 = 0.15
        // Plus regularization: rank is 2 (0-indexed 1), so (2 - 5)^+ = 0
        // Total: 0.6 + 0.15 + 0 = 0.75
        assert!((score - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_raps_with_regularization() {
        let config = RapsConfig {
            lambda: 0.1,
            k_reg: 2,
            randomize_ties: false,
        };
        let scorer = RapsScorer::new(config);

        // 5-class example where true label is ranked low
        let softmax = vec![0.4, 0.25, 0.15, 0.12, 0.08];
        let true_label = 4; // Ranked 5th (0-indexed rank 4)
        let u = 0.5;

        let score = scorer.score(&softmax, true_label, u);

        // cumsum = 0.4 + 0.25 + 0.15 + 0.12 = 0.92
        // base = 0.92 + 0.5 * 0.08 = 0.96
        // reg = 0.1 * (5 - 2) = 0.3
        // total = 0.96 + 0.3 = 1.26
        assert!((score - 1.26).abs() < 1e-6);
    }

    #[test]
    fn test_raps_top_class() {
        let config = RapsConfig::default();
        let scorer = RapsScorer::new(config);

        let softmax = vec![0.7, 0.2, 0.1];
        let true_label = 0; // Top class
        let u = 0.5;

        let score = scorer.score(&softmax, true_label, u);

        // cumsum = 0 (no classes ranked higher)
        // base = 0 + 0.5 * 0.7 = 0.35
        // reg = 0 (rank 1 < k_reg = 5)
        // total = 0.35
        assert!((score - 0.35).abs() < 1e-6);
    }

    #[test]
    fn test_raps_batch() {
        let config = RapsConfig::default();
        let scorer = RapsScorer::new(config);

        let batch = vec![
            vec![0.6, 0.3, 0.1],
            vec![0.4, 0.4, 0.2],
            vec![0.5, 0.3, 0.2],
        ];
        let labels = vec![0, 1, 2];
        let u_values = vec![0.5, 0.5, 0.5];

        let scores = scorer.score_batch(&batch, &labels, &u_values);

        assert_eq!(scores.len(), 3);
        assert!(scores.iter().all(|&s| s >= 0.0));
    }

    #[test]
    fn test_raps_deterministic_with_u_05() {
        let config = RapsConfig {
            randomize_ties: false,
            ..Default::default()
        };
        let scorer = RapsScorer::new(config);

        let softmax = vec![0.6, 0.3, 0.1];
        let true_label = 1;

        let score1 = scorer.score(&softmax, true_label, 0.5);
        let score2 = scorer.score(&softmax, true_label, 0.5);

        assert_eq!(score1, score2);
    }

    #[test]
    fn test_raps_monotonicity() {
        // Score should increase as true label's probability decreases
        let config = RapsConfig::default();
        let scorer = RapsScorer::new(config);

        let softmax1 = vec![0.5, 0.3, 0.2];
        let softmax2 = vec![0.6, 0.25, 0.15];

        let score1 = scorer.score(&softmax1, 1, 0.5); // true_label has prob 0.3
        let score2 = scorer.score(&softmax2, 1, 0.5); // true_label has prob 0.25

        assert!(score2 > score1, "Lower probability should give higher score");
    }

    #[test]
    #[should_panic]
    fn test_raps_invalid_label() {
        let config = RapsConfig::default();
        let scorer = RapsScorer::new(config);

        let softmax = vec![0.6, 0.3, 0.1];
        let _ = scorer.score(&softmax, 5, 0.5); // Out of bounds
    }
}
