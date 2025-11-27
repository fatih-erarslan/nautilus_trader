//! Mondrian Conformal Prediction (Vovk et al., 2012; Romano et al., 2020)
//!
//! Provides group-conditional validity: P(Y ∈ C(X) | G(X) = g) ≥ 1 - α
//! for each group g in the partition.
//!
//! Uses separate calibration per group to ensure coverage within each stratum,
//! enabling fairness-aware prediction sets.
//!
//! # Mathematical Foundation
//!
//! For partition {G₁, ..., Gₖ}, computes separate quantiles:
//! τ_g = Quantile_{(1-α)(1+1/|G_g|)}({s_i : i ∈ G_g})
//!
//! where s_i is the nonconformity score for calibration sample i.

use std::collections::HashMap;
use super::NonconformityScore;

/// Group identifier (can be categorical or discretized continuous)
pub type GroupId = u64;

/// Configuration for Mondrian conformal predictor
#[derive(Clone, Debug)]
pub struct MondrianConfig {
    /// Miscoverage level α (typically 0.1 for 90% coverage)
    pub alpha: f32,
    /// Minimum samples per group for valid calibration
    pub min_group_size: usize,
    /// Fallback to marginal calibration when group too small
    pub fallback_to_marginal: bool,
    /// Use conservative (ceiling) or liberal (floor) quantile index
    pub conservative: bool,
}

impl Default for MondrianConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            min_group_size: 30,
            fallback_to_marginal: true,
            conservative: true,
        }
    }
}

/// Mondrian conformal calibrator with group-conditional guarantees
///
/// # Type Parameters
/// * `S` - Nonconformity scorer implementing the scoring function
///
/// # Examples
///
/// ```ignore
/// let config = MondrianConfig::default();
/// let scorer = RapsScorer::new(RapsConfig::default());
/// let mut calibrator = MondrianCalibrator::new(config, scorer);
///
/// calibrator.calibrate(&predictions, &labels, &groups, &u_values);
/// let pred_set = calibrator.predict_set(&new_prediction, group_id);
/// ```
pub struct MondrianCalibrator<S: NonconformityScore> {
    config: MondrianConfig,
    scorer: S,
    /// Per-group calibration thresholds τ_g
    group_thresholds: HashMap<GroupId, f32>,
    /// Marginal threshold (fallback for unseen/small groups)
    marginal_threshold: Option<f32>,
    /// Group sample counts for diagnostics
    group_counts: HashMap<GroupId, usize>,
}

impl<S: NonconformityScore> MondrianCalibrator<S> {
    /// Create new Mondrian calibrator
    ///
    /// # Arguments
    /// * `config` - Configuration parameters
    /// * `scorer` - Nonconformity score function
    pub fn new(config: MondrianConfig, scorer: S) -> Self {
        Self {
            config,
            scorer,
            group_thresholds: HashMap::new(),
            marginal_threshold: None,
            group_counts: HashMap::new(),
        }
    }

    /// Calibrate with group assignments
    ///
    /// Computes separate thresholds for each group in the partition.
    ///
    /// # Arguments
    /// * `predictions` - Model predictions (softmax probabilities), shape (n, num_classes)
    /// * `labels` - True labels, shape (n,)
    /// * `groups` - Group assignments for each sample, shape (n,)
    /// * `u_values` - Random uniforms [0,1] for tie-breaking, shape (n,)
    ///
    /// # Panics
    /// If array lengths don't match or if no samples provided
    ///
    /// # Mathematical Details
    ///
    /// For each group g:
    /// 1. Collect scores: S_g = {s_i : G(x_i) = g}
    /// 2. Compute quantile: τ_g = Quantile_{q}(S_g) where q = (1-α)(1 + 1/|S_g|)
    /// 3. Store threshold for prediction time
    pub fn calibrate(
        &mut self,
        predictions: &[Vec<f32>],
        labels: &[usize],
        groups: &[GroupId],
        u_values: &[f32],
    ) {
        let n = predictions.len();
        assert_eq!(n, labels.len(), "Predictions and labels length mismatch");
        assert_eq!(n, groups.len(), "Predictions and groups length mismatch");
        assert_eq!(n, u_values.len(), "Predictions and u_values length mismatch");
        assert!(n > 0, "Cannot calibrate with zero samples");

        // Group samples by their group ID
        let mut group_scores: HashMap<GroupId, Vec<f32>> = HashMap::new();
        let mut all_scores = Vec::with_capacity(n);

        for i in 0..n {
            let score = self.scorer.score(&predictions[i], labels[i], u_values[i]);
            all_scores.push(score);

            group_scores.entry(groups[i])
                .or_insert_with(Vec::new)
                .push(score);
        }

        // Compute marginal threshold (pooled across all groups)
        self.marginal_threshold = Some(
            Self::compute_quantile(&all_scores, self.config.alpha, self.config.conservative)
        );

        // Compute per-group thresholds
        for (group_id, scores) in group_scores {
            self.group_counts.insert(group_id, scores.len());

            if scores.len() >= self.config.min_group_size {
                let threshold = Self::compute_quantile(&scores, self.config.alpha, self.config.conservative);
                self.group_thresholds.insert(group_id, threshold);
            }
        }
    }

    /// Compute (1-α)(1 + 1/n) quantile from scores
    ///
    /// # Mathematical Formula
    ///
    /// q = ⌈(n+1)(1-α)⌉ for conservative (default)
    /// q = ⌊(n+1)(1-α)⌋ for liberal
    ///
    /// This ensures finite-sample validity under exchangeability.
    ///
    /// # Arguments
    /// * `scores` - Nonconformity scores
    /// * `alpha` - Miscoverage level
    /// * `conservative` - Use ceiling (true) or floor (false)
    ///
    /// # Returns
    /// Threshold value τ
    fn compute_quantile(scores: &[f32], alpha: f32, conservative: bool) -> f32 {
        let mut sorted = scores.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let level = (1.0 - alpha) * (1.0 + 1.0 / n as f32);

        let idx = if conservative {
            ((level * n as f32).ceil() as usize).min(n - 1)
        } else {
            ((level * n as f32).floor() as usize).min(n - 1)
        };

        sorted[idx]
    }

    /// Get calibrated threshold for a group
    ///
    /// Returns group-specific threshold if available and group size >= min_group_size,
    /// otherwise returns marginal threshold if fallback enabled.
    ///
    /// # Arguments
    /// * `group` - Group identifier
    ///
    /// # Returns
    /// Calibrated threshold τ_g or τ_marginal
    ///
    /// # Panics
    /// If group not seen during calibration and fallback disabled
    pub fn get_threshold(&self, group: GroupId) -> f32 {
        if let Some(&threshold) = self.group_thresholds.get(&group) {
            threshold
        } else if self.config.fallback_to_marginal {
            self.marginal_threshold.expect("Must calibrate before prediction")
        } else {
            panic!("No threshold for group {} and fallback disabled", group);
        }
    }

    /// Produce conformal prediction set for new sample
    ///
    /// Returns set C(x) such that P(Y ∈ C(X) | G(X) = g) ≥ 1 - α
    ///
    /// # Arguments
    /// * `prediction` - Softmax probabilities for new sample
    /// * `group` - Group assignment for new sample
    ///
    /// # Returns
    /// Vector of class indices in the prediction set
    ///
    /// # Algorithm
    ///
    /// 1. Get group threshold τ_g
    /// 2. Sort classes by descending probability
    /// 3. Include classes until cumulative probability >= τ_g
    pub fn predict_set(
        &self,
        prediction: &[f32],
        group: GroupId,
    ) -> Vec<usize> {
        let threshold = self.get_threshold(group);

        // Sort classes by descending probability
        let mut indices: Vec<usize> = (0..prediction.len()).collect();
        indices.sort_by(|&a, &b| {
            prediction[b].partial_cmp(&prediction[a]).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut cumsum = 0.0;
        let mut pred_set = Vec::new();

        // Accumulate until threshold reached
        for &idx in &indices {
            pred_set.push(idx);
            cumsum += prediction[idx];
            if cumsum >= threshold {
                break;
            }
        }

        pred_set
    }

    /// Get group statistics for diagnostics
    ///
    /// Returns map of group_id -> (threshold, sample_count)
    pub fn get_group_statistics(&self) -> HashMap<GroupId, (f32, usize)> {
        self.group_thresholds.iter()
            .map(|(&g, &t)| (g, (t, *self.group_counts.get(&g).unwrap_or(&0))))
            .collect()
    }

    /// Get marginal threshold
    pub fn get_marginal_threshold(&self) -> Option<f32> {
        self.marginal_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock scorer for testing
    #[derive(Clone)]
    struct MockScorer;

    impl NonconformityScore for MockScorer {
        fn score(&self, prediction: &[f32], label: usize, _u: f32) -> f32 {
            // Simple score: 1 - p(label)
            1.0 - prediction[label]
        }
    }

    #[test]
    fn test_mondrian_basic_calibration() {
        let config = MondrianConfig {
            alpha: 0.1,
            min_group_size: 2,
            fallback_to_marginal: true,
            conservative: true,
        };
        let scorer = MockScorer;
        let mut calibrator = MondrianCalibrator::new(config, scorer);

        // Create test data: 2 groups with different distributions
        let predictions = vec![
            vec![0.7, 0.2, 0.1],  // Group 0, label 0
            vec![0.6, 0.3, 0.1],  // Group 0, label 0
            vec![0.1, 0.8, 0.1],  // Group 1, label 1
            vec![0.1, 0.7, 0.2],  // Group 1, label 1
        ];
        let labels = vec![0, 0, 1, 1];
        let groups = vec![0, 0, 1, 1];
        let u_values = vec![0.5, 0.5, 0.5, 0.5];

        calibrator.calibrate(&predictions, &labels, &groups, &u_values);

        // Verify thresholds computed
        assert!(calibrator.get_threshold(0) > 0.0);
        assert!(calibrator.get_threshold(1) > 0.0);
        assert!(calibrator.get_marginal_threshold().is_some());
    }

    #[test]
    fn test_mondrian_predict_set() {
        let config = MondrianConfig::default();
        let scorer = MockScorer;
        let mut calibrator = MondrianCalibrator::new(config, scorer);

        // Calibrate
        let predictions = vec![
            vec![0.8, 0.1, 0.1],
            vec![0.7, 0.2, 0.1],
            vec![0.1, 0.8, 0.1],
        ];
        let labels = vec![0, 0, 1];
        let groups = vec![0, 0, 1];
        let u_values = vec![0.5, 0.5, 0.5];

        calibrator.calibrate(&predictions, &labels, &groups, &u_values);

        // Test prediction
        let new_pred = vec![0.6, 0.3, 0.1];
        let pred_set = calibrator.predict_set(&new_pred, 0);

        assert!(!pred_set.is_empty(), "Prediction set should not be empty");
        assert!(pred_set.contains(&0) || pred_set.contains(&1),
                "Should contain high-probability classes");
    }

    #[test]
    fn test_mondrian_group_coverage() {
        // Test that different groups get different thresholds
        let config = MondrianConfig {
            alpha: 0.1,
            min_group_size: 3,
            fallback_to_marginal: false,
            conservative: true,
        };
        let scorer = MockScorer;
        let mut calibrator = MondrianCalibrator::new(config, scorer);

        // Group 0: high confidence predictions
        // Group 1: low confidence predictions
        let predictions = vec![
            vec![0.9, 0.05, 0.05],  // Group 0
            vec![0.85, 0.10, 0.05], // Group 0
            vec![0.88, 0.07, 0.05], // Group 0
            vec![0.4, 0.35, 0.25],  // Group 1
            vec![0.45, 0.30, 0.25], // Group 1
            vec![0.42, 0.33, 0.25], // Group 1
        ];
        let labels = vec![0, 0, 0, 0, 0, 0];
        let groups = vec![0, 0, 0, 1, 1, 1];
        let u_values = vec![0.5; 6];

        calibrator.calibrate(&predictions, &labels, &groups, &u_values);

        let t0 = calibrator.get_threshold(0);
        let t1 = calibrator.get_threshold(1);

        // Group 0 should have lower threshold (higher confidence)
        assert!(t0 < t1, "High-confidence group should have lower threshold");
    }

    #[test]
    fn test_mondrian_fallback_to_marginal() {
        let config = MondrianConfig {
            alpha: 0.1,
            min_group_size: 10, // High threshold
            fallback_to_marginal: true,
            conservative: true,
        };
        let scorer = MockScorer;
        let mut calibrator = MondrianCalibrator::new(config, scorer);

        // Small group that falls below min_group_size
        let predictions = vec![
            vec![0.7, 0.2, 0.1],
            vec![0.6, 0.3, 0.1],
        ];
        let labels = vec![0, 0];
        let groups = vec![0, 0];
        let u_values = vec![0.5, 0.5];

        calibrator.calibrate(&predictions, &labels, &groups, &u_values);

        // Should fallback to marginal
        let threshold = calibrator.get_threshold(0);
        let marginal = calibrator.get_marginal_threshold().unwrap();

        assert_eq!(threshold, marginal,
                   "Small group should use marginal threshold");
    }

    #[test]
    fn test_quantile_computation() {
        let scores = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let alpha = 0.1;

        let conservative = MondrianCalibrator::<MockScorer>::compute_quantile(
            &scores, alpha, true
        );
        let liberal = MondrianCalibrator::<MockScorer>::compute_quantile(
            &scores, alpha, false
        );

        assert!(conservative >= liberal,
                "Conservative quantile should be >= liberal");
        assert!(conservative <= *scores.last().unwrap(),
                "Quantile should not exceed max score");
    }
}
