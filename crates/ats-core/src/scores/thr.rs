//! Threshold-based Nonconformity Scores
//!
//! Simple threshold-based scoring function:
//!
//! ```text
//! s(x,y) = 1 - π̂_y
//! ```
//!
//! This is the simplest possible nonconformity measure, equivalent to
//! using the inverse of the predicted probability for the true class.

use crate::NonconformityScorer;

/// Threshold scorer configuration
#[derive(Clone, Debug)]
pub struct ThresholdConfig {
    /// Whether to use randomization (unused for THR, kept for consistency)
    pub randomize: bool,
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self { randomize: false }
    }
}

/// Threshold-based nonconformity scorer
///
/// # Example
///
/// ```
/// use ats_core::scores::{ThresholdScorer, NonconformityScorer};
///
/// let scorer = ThresholdScorer::default();
///
/// let softmax = vec![0.6, 0.3, 0.1];
/// let true_label = 1;
///
/// let score = scorer.score(&softmax, true_label, 0.0); // u is ignored
/// assert!((score - 0.7).abs() < 1e-6); // 1 - 0.3 = 0.7
/// ```
pub struct ThresholdScorer {
    config: ThresholdConfig,
}

impl ThresholdScorer {
    /// Create a new threshold scorer
    pub fn new(config: ThresholdConfig) -> Self {
        Self { config }
    }

    /// Create threshold scorer with default configuration
    pub fn default() -> Self {
        Self::new(ThresholdConfig::default())
    }

    /// Get current configuration
    pub fn config(&self) -> &ThresholdConfig {
        &self.config
    }

    /// Compute threshold score directly (static function)
    ///
    /// # Arguments
    /// * `softmax_probs` - Softmax probabilities for all classes
    /// * `true_label` - Index of the true class
    ///
    /// # Returns
    /// Score s(x,y) = 1 - π̂_y
    #[inline]
    pub fn threshold_score(softmax_probs: &[f32], true_label: usize) -> f32 {
        debug_assert!(
            true_label < softmax_probs.len(),
            "true_label out of bounds"
        );
        1.0 - softmax_probs[true_label]
    }
}

impl NonconformityScorer for ThresholdScorer {
    #[inline]
    fn score(&self, softmax_probs: &[f32], true_label: usize, _u: f32) -> f32 {
        // u is ignored for threshold scorer
        Self::threshold_score(softmax_probs, true_label)
    }

    fn score_batch(
        &self,
        softmax_batch: &[Vec<f32>],
        labels: &[usize],
        _u_values: &[f32],
    ) -> Vec<f32> {
        debug_assert_eq!(softmax_batch.len(), labels.len());

        softmax_batch
            .iter()
            .zip(labels.iter())
            .map(|(probs, &label)| Self::threshold_score(probs, label))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_basic() {
        let scorer = ThresholdScorer::default();

        let softmax = vec![0.6, 0.3, 0.1];
        let true_label = 1;

        let score = scorer.score(&softmax, true_label, 0.0);

        // 1 - 0.3 = 0.7
        assert!((score - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_threshold_high_confidence() {
        let scorer = ThresholdScorer::default();

        let softmax = vec![0.9, 0.05, 0.05];
        let true_label = 0;

        let score = scorer.score(&softmax, true_label, 0.0);

        // 1 - 0.9 = 0.1 (low score for high confidence)
        assert!((score - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_threshold_low_confidence() {
        let scorer = ThresholdScorer::default();

        let softmax = vec![0.4, 0.4, 0.2];
        let true_label = 2;

        let score = scorer.score(&softmax, true_label, 0.0);

        // 1 - 0.2 = 0.8 (high score for low confidence)
        assert!((score - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_threshold_u_ignored() {
        let scorer = ThresholdScorer::default();

        let softmax = vec![0.6, 0.3, 0.1];
        let true_label = 1;

        let score1 = scorer.score(&softmax, true_label, 0.0);
        let score2 = scorer.score(&softmax, true_label, 0.5);
        let score3 = scorer.score(&softmax, true_label, 1.0);

        assert_eq!(score1, score2);
        assert_eq!(score2, score3);
    }

    #[test]
    fn test_threshold_batch() {
        let scorer = ThresholdScorer::default();

        let batch = vec![
            vec![0.6, 0.3, 0.1],
            vec![0.5, 0.3, 0.2],
            vec![0.4, 0.4, 0.2],
        ];
        let labels = vec![0, 1, 2];
        let u_values = vec![0.0, 0.0, 0.0];

        let scores = scorer.score_batch(&batch, &labels, &u_values);

        assert_eq!(scores.len(), 3);
        assert!((scores[0] - 0.4).abs() < 1e-6); // 1 - 0.6
        assert!((scores[1] - 0.7).abs() < 1e-6); // 1 - 0.3
        assert!((scores[2] - 0.8).abs() < 1e-6); // 1 - 0.2
    }

    #[test]
    fn test_threshold_static_function() {
        let softmax = vec![0.6, 0.3, 0.1];

        let score = ThresholdScorer::threshold_score(&softmax, 1);

        assert!((score - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_threshold_bounds() {
        let scorer = ThresholdScorer::default();

        // Extreme confidence
        let softmax1 = vec![1.0, 0.0, 0.0];
        let score1 = scorer.score(&softmax1, 0, 0.0);
        assert!((score1 - 0.0).abs() < 1e-6);

        // Uniform distribution
        let softmax2 = vec![0.333, 0.333, 0.334];
        let score2 = scorer.score(&softmax2, 0, 0.0);
        assert!((score2 - 0.667).abs() < 0.01);
    }
}
