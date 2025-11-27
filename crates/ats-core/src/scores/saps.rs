//! Sorted Adaptive Prediction Sets (SAPS)
//!
//! Size-adaptive variant of APS that penalizes larger prediction sets.
//!
//! # Mathematical Definition
//!
//! ```text
//! s(x,y) = Σ_{j: π̂_j > π̂_y} π̂_j + u·π̂_y + penalty(set_size)
//! ```
//!
//! Where the penalty function encourages smaller sets while maintaining coverage.

use crate::NonconformityScorer;

#[derive(Clone, Debug)]
pub struct SapsConfig {
    /// Random tie-breaking for equal probabilities
    pub randomize_ties: bool,

    /// Penalty coefficient for set size
    pub size_penalty: f32,
}

impl Default for SapsConfig {
    fn default() -> Self {
        Self {
            randomize_ties: true,
            size_penalty: 0.01,
        }
    }
}

/// SAPS nonconformity scorer
pub struct SapsScorer {
    config: SapsConfig,
}

impl SapsScorer {
    /// Create a new SAPS scorer with the given configuration
    pub fn new(config: SapsConfig) -> Self {
        Self { config }
    }

    /// Create SAPS scorer with default configuration
    pub fn default() -> Self {
        Self::new(SapsConfig::default())
    }

    /// Get current configuration
    pub fn config(&self) -> &SapsConfig {
        &self.config
    }
}

impl NonconformityScorer for SapsScorer {
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
        let base_score = cumsum + u * true_prob;

        // Add size penalty proportional to estimated set size
        // Estimated set size is approximately the number of classes
        // with cumulative probability up to the threshold
        let size_penalty = self.config.size_penalty * (rank + 1) as f32;

        base_score + size_penalty
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
    fn test_saps_basic() {
        let config = SapsConfig::default();
        let scorer = SapsScorer::new(config);

        let softmax = vec![0.6, 0.3, 0.1];
        let true_label = 1;
        let u = 0.5;

        let score = scorer.score(&softmax, true_label, u);

        // cumsum = 0.6, base = 0.6 + 0.5 * 0.3 = 0.75
        // penalty = 0.01 * 2 = 0.02 (rank is 1, so rank+1 = 2)
        // total = 0.75 + 0.02 = 0.77
        assert!((score - 0.77).abs() < 1e-6);
    }

    #[test]
    fn test_saps_size_penalty_effect() {
        let config1 = SapsConfig {
            size_penalty: 0.0,
            ..Default::default()
        };
        let config2 = SapsConfig {
            size_penalty: 0.1,
            ..Default::default()
        };

        let scorer1 = SapsScorer::new(config1);
        let scorer2 = SapsScorer::new(config2);

        let softmax = vec![0.6, 0.3, 0.1];
        let true_label = 1;
        let u = 0.5;

        let score1 = scorer1.score(&softmax, true_label, u);
        let score2 = scorer2.score(&softmax, true_label, u);

        assert!(score2 > score1, "Higher penalty should increase score");
    }

    #[test]
    fn test_saps_batch() {
        let config = SapsConfig::default();
        let scorer = SapsScorer::new(config);

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
}
