//! Kernel-based Localized Conformal Prediction (Kandinsky Method)
//!
//! Provides approximate conditional coverage through kernel weighting,
//! achieving local validity near query points in feature space.
//!
//! # Mathematical Foundation
//!
//! Uses kernel-weighted quantiles:
//! τ(x) = Quantile_{w(x)}({s_i}_{i=1}^n)
//!
//! where w_i(x) = K(x, x_i) and K is a kernel function (Gaussian, Epanechnikov, etc.)
//!
//! # References
//! - Barber et al. (2023): "Conformal Prediction Beyond Exchangeability"
//! - Lei & Wasserman (2014): "Distribution-Free Prediction Bands"

use super::NonconformityScore;

/// Kernel function types
#[derive(Clone, Debug, PartialEq)]
pub enum KernelType {
    /// Gaussian/RBF kernel: K(x,y) = exp(-||x-y||²/(2h²))
    Gaussian,
    /// Epanechnikov kernel: K(x,y) = max(0, 1 - ||x-y||²/h²)
    Epanechnikov,
    /// Tricube kernel: K(x,y) = (1 - ||x-y||³/h³)³ for ||x-y|| < h
    Tricube,
}

/// Configuration for Kandinsky conformal predictor
#[derive(Clone, Debug)]
pub struct KandinskyConfig {
    /// Miscoverage level α
    pub alpha: f32,
    /// Kernel bandwidth h (critical tuning parameter)
    pub bandwidth: f32,
    /// Kernel type
    pub kernel_type: KernelType,
    /// Minimum effective sample size (sum of weights)
    pub min_effective_samples: f32,
    /// Fallback to marginal when effective samples too low
    pub fallback_to_marginal: bool,
}

impl Default for KandinskyConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            bandwidth: 1.0,
            kernel_type: KernelType::Gaussian,
            min_effective_samples: 30.0,
            fallback_to_marginal: true,
        }
    }
}

/// Kandinsky conformal calibrator with kernel-based localization
///
/// # Type Parameters
/// * `S` - Nonconformity scorer
///
/// # Examples
///
/// ```ignore
/// let config = KandinskyConfig {
///     bandwidth: 0.5,
///     kernel_type: KernelType::Gaussian,
///     ..Default::default()
/// };
/// let scorer = RapsScorer::new(RapsConfig::default());
/// let mut calibrator = KandinskyCalibrator::new(config, scorer);
///
/// calibrator.calibrate(&predictions, &labels, &features, &u_values);
/// let pred_set = calibrator.predict_set(&new_prediction, &new_features);
/// ```
pub struct KandinskyCalibrator<S: NonconformityScore> {
    config: KandinskyConfig,
    scorer: S,
    /// Calibration features for kernel weighting
    calibration_features: Vec<Vec<f32>>,
    /// Calibration nonconformity scores
    calibration_scores: Vec<f32>,
    /// Marginal threshold (fallback)
    marginal_threshold: Option<f32>,
}

impl<S: NonconformityScore> KandinskyCalibrator<S> {
    /// Create new Kandinsky calibrator
    ///
    /// # Arguments
    /// * `config` - Configuration parameters
    /// * `scorer` - Nonconformity score function
    pub fn new(config: KandinskyConfig, scorer: S) -> Self {
        Self {
            config,
            scorer,
            calibration_features: Vec::new(),
            calibration_scores: Vec::new(),
            marginal_threshold: None,
        }
    }

    /// Calibrate with feature vectors for kernel weighting
    ///
    /// # Arguments
    /// * `predictions` - Model predictions (softmax probabilities)
    /// * `labels` - True labels
    /// * `features` - Feature vectors for kernel weighting
    /// * `u_values` - Random uniforms for tie-breaking
    ///
    /// # Panics
    /// If array lengths don't match or features have inconsistent dimensions
    pub fn calibrate(
        &mut self,
        predictions: &[Vec<f32>],
        labels: &[usize],
        features: &[Vec<f32>],
        u_values: &[f32],
    ) {
        let n = predictions.len();
        assert_eq!(n, labels.len());
        assert_eq!(n, features.len());
        assert_eq!(n, u_values.len());
        assert!(n > 0, "Cannot calibrate with zero samples");

        // Verify feature dimension consistency
        if !features.is_empty() {
            let dim = features[0].len();
            assert!(features.iter().all(|f| f.len() == dim),
                   "All feature vectors must have same dimension");
        }

        // Compute and store scores
        self.calibration_scores.clear();
        for i in 0..n {
            let score = self.scorer.score(&predictions[i], labels[i], u_values[i]);
            self.calibration_scores.push(score);
        }

        // Store features for kernel weighting at prediction time
        self.calibration_features = features.to_vec();

        // Compute marginal threshold (uniform weights)
        let uniform_weights = vec![1.0 / n as f32; n];
        self.marginal_threshold = Some(
            self.weighted_quantile(&self.calibration_scores, &uniform_weights, 1.0 - self.config.alpha)
        );
    }

    /// Compute kernel-weighted quantile for query point
    ///
    /// # Algorithm
    ///
    /// 1. Compute kernel weights: w_i = K(x_query, x_i^{cal})
    /// 2. Normalize: w_i ← w_i / Σw_j
    /// 3. Sort scores with weights
    /// 4. Find weighted quantile: first s where Σ{w_j : s_j ≤ s} ≥ (1-α)
    ///
    /// # Arguments
    /// * `query_features` - Feature vector for new sample
    /// * `alpha` - Miscoverage level
    ///
    /// # Returns
    /// Localized threshold τ(x)
    fn compute_weighted_threshold(
        &self,
        query_features: &[f32],
        alpha: f32,
    ) -> f32 {
        // Compute kernel weights for all calibration points
        let weights: Vec<f32> = self.calibration_features.iter()
            .map(|x| self.kernel(query_features, x))
            .collect();

        // Check effective sample size
        let sum_weights: f32 = weights.iter().sum();
        let effective_n = sum_weights.powi(2) / weights.iter().map(|w| w.powi(2)).sum::<f32>();

        if effective_n < self.config.min_effective_samples && self.config.fallback_to_marginal {
            return self.marginal_threshold.expect("Must calibrate before prediction");
        }

        // Normalize weights
        let normalized: Vec<f32> = weights.iter().map(|w| w / sum_weights).collect();

        // Compute weighted quantile
        self.weighted_quantile(&self.calibration_scores, &normalized, 1.0 - alpha)
    }

    /// Compute kernel function K(x, y)
    ///
    /// # Arguments
    /// * `x` - First point
    /// * `y` - Second point
    ///
    /// # Returns
    /// Kernel value K(x, y) ≥ 0
    fn kernel(&self, x: &[f32], y: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len(), "Feature dimensions must match");

        let sq_dist: f32 = x.iter().zip(y.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        let h = self.config.bandwidth;

        match self.config.kernel_type {
            KernelType::Gaussian => {
                // K(x,y) = exp(-||x-y||²/(2h²))
                (-sq_dist / (2.0 * h.powi(2))).exp()
            }
            KernelType::Epanechnikov => {
                // K(x,y) = max(0, 1 - ||x-y||²/h²)
                (1.0 - sq_dist / h.powi(2)).max(0.0)
            }
            KernelType::Tricube => {
                // K(x,y) = (1 - (||x-y||/h)³)³ for ||x-y|| < h
                let dist = sq_dist.sqrt();
                if dist >= h {
                    0.0
                } else {
                    let u = dist / h;
                    (1.0 - u.powi(3)).powi(3)
                }
            }
        }
    }

    /// Compute weighted quantile
    ///
    /// # Algorithm (Weighted Quantile)
    ///
    /// 1. Create pairs (score, weight) and sort by score
    /// 2. Accumulate weights until cumulative weight ≥ quantile_level
    /// 3. Return corresponding score
    ///
    /// # Arguments
    /// * `scores` - Values to compute quantile over
    /// * `weights` - Normalized weights (must sum to 1)
    /// * `quantile` - Quantile level (e.g., 0.9 for 90th percentile)
    ///
    /// # Returns
    /// Weighted quantile value
    fn weighted_quantile(
        &self,
        scores: &[f32],
        weights: &[f32],
        quantile: f32,
    ) -> f32 {
        assert_eq!(scores.len(), weights.len());
        assert!(quantile >= 0.0 && quantile <= 1.0);

        if scores.is_empty() {
            return 0.0;
        }

        // Create indexed pairs and sort by score
        let mut pairs: Vec<(f32, f32)> = scores.iter()
            .zip(weights.iter())
            .map(|(&s, &w)| (s, w))
            .collect();

        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Accumulate weights until quantile reached
        let mut cumulative_weight = 0.0;
        let mut last_score = pairs.last().map(|(s, _)| *s).unwrap_or(0.0);
        for (score, weight) in &pairs {
            cumulative_weight += weight;
            if cumulative_weight >= quantile {
                return *score;
            }
        }

        // Return maximum if we didn't reach quantile (numerical precision)
        last_score
    }

    /// Produce conformal prediction set for new sample
    ///
    /// # Arguments
    /// * `prediction` - Softmax probabilities
    /// * `features` - Feature vector for kernel weighting
    ///
    /// # Returns
    /// Vector of class indices in prediction set
    pub fn predict_set(
        &self,
        prediction: &[f32],
        features: &[f32],
    ) -> Vec<usize> {
        let threshold = self.compute_weighted_threshold(features, self.config.alpha);

        // Sort classes by descending probability
        let mut indices: Vec<usize> = (0..prediction.len()).collect();
        indices.sort_by(|&a, &b| {
            prediction[b].partial_cmp(&prediction[a]).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut cumsum = 0.0;
        let mut pred_set = Vec::new();

        for &idx in &indices {
            pred_set.push(idx);
            cumsum += prediction[idx];
            if cumsum >= threshold {
                break;
            }
        }

        pred_set
    }

    /// Get effective sample size for query point
    ///
    /// Effective sample size: n_eff = (Σw_i)² / Σ(w_i²)
    ///
    /// # Arguments
    /// * `query_features` - Query point
    ///
    /// # Returns
    /// Effective sample size
    pub fn effective_sample_size(&self, query_features: &[f32]) -> f32 {
        let weights: Vec<f32> = self.calibration_features.iter()
            .map(|x| self.kernel(query_features, x))
            .collect();

        let sum_weights: f32 = weights.iter().sum();
        let sum_sq_weights: f32 = weights.iter().map(|w| w.powi(2)).sum();

        // Handle edge case: when all weights are 0, effective sample size is 0
        if sum_sq_weights < f32::EPSILON {
            return 0.0;
        }

        sum_weights.powi(2) / sum_sq_weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct MockScorer;

    impl NonconformityScore for MockScorer {
        fn score(&self, prediction: &[f32], label: usize, _u: f32) -> f32 {
            1.0 - prediction[label]
        }
    }

    #[test]
    fn test_gaussian_kernel() {
        let config = KandinskyConfig {
            bandwidth: 1.0,
            kernel_type: KernelType::Gaussian,
            ..Default::default()
        };
        let calibrator = KandinskyCalibrator::new(config, MockScorer);

        // Same point should give kernel value 1.0
        let x = vec![1.0, 2.0, 3.0];
        let k = calibrator.kernel(&x, &x);
        assert!((k - 1.0).abs() < 1e-6, "K(x,x) should be 1.0");

        // Distant points should give smaller kernel value
        let y = vec![10.0, 20.0, 30.0];
        let k_far = calibrator.kernel(&x, &y);
        assert!(k_far < 0.01, "K(x,y) should be small for distant points");
    }

    #[test]
    fn test_epanechnikov_kernel() {
        let config = KandinskyConfig {
            bandwidth: 2.0,
            kernel_type: KernelType::Epanechnikov,
            ..Default::default()
        };
        let calibrator = KandinskyCalibrator::new(config, MockScorer);

        let x = vec![0.0, 0.0];
        let y = vec![1.0, 0.0]; // Distance = 1.0

        let k = calibrator.kernel(&x, &y);
        // K(x,y) = max(0, 1 - 1²/2²) = 0.75
        assert!((k - 0.75).abs() < 1e-6);

        // Beyond bandwidth
        let z = vec![3.0, 0.0]; // Distance = 3.0 > h=2.0
        let k_zero = calibrator.kernel(&x, &z);
        assert_eq!(k_zero, 0.0, "Epanechnikov kernel should be 0 beyond bandwidth");
    }

    #[test]
    fn test_tricube_kernel() {
        let config = KandinskyConfig {
            bandwidth: 1.0,
            kernel_type: KernelType::Tricube,
            ..Default::default()
        };
        let calibrator = KandinskyCalibrator::new(config, MockScorer);

        let x = vec![0.0];
        let y = vec![0.5]; // Distance = 0.5

        let k = calibrator.kernel(&x, &y);
        // K(0.5) = (1 - 0.5³)³ = (1 - 0.125)³ = 0.875³ ≈ 0.669
        assert!((k - 0.669).abs() < 0.01);
    }

    #[test]
    fn test_weighted_quantile() {
        let config = KandinskyConfig::default();
        let calibrator = KandinskyCalibrator::new(config, MockScorer);

        let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let uniform_weights = vec![0.2, 0.2, 0.2, 0.2, 0.2];

        let q50 = calibrator.weighted_quantile(&scores, &uniform_weights, 0.5);
        assert_eq!(q50, 3.0, "Median of [1,2,3,4,5] should be 3");

        let q90 = calibrator.weighted_quantile(&scores, &uniform_weights, 0.9);
        assert!(q90 >= 4.0 && q90 <= 5.0, "90th percentile should be in [4,5]");

        // Test weighted quantile with non-uniform weights
        let skewed_weights = vec![0.5, 0.1, 0.1, 0.1, 0.2];
        let q50_weighted = calibrator.weighted_quantile(&scores, &skewed_weights, 0.5);
        // With weight 0.5 on score 1.0, median should be 1.0
        assert_eq!(q50_weighted, 1.0);
    }

    #[test]
    fn test_kandinsky_calibration() {
        let config = KandinskyConfig {
            bandwidth: 0.5,
            ..Default::default()
        };
        let scorer = MockScorer;
        let mut calibrator = KandinskyCalibrator::new(config, scorer);

        let predictions = vec![
            vec![0.7, 0.2, 0.1],
            vec![0.6, 0.3, 0.1],
            vec![0.1, 0.8, 0.1],
        ];
        let labels = vec![0, 0, 1];
        let features = vec![
            vec![1.0, 0.0],
            vec![1.1, 0.1],
            vec![5.0, 5.0],
        ];
        let u_values = vec![0.5, 0.5, 0.5];

        calibrator.calibrate(&predictions, &labels, &features, &u_values);

        assert_eq!(calibrator.calibration_scores.len(), 3);
        assert!(calibrator.marginal_threshold.is_some());
    }

    #[test]
    fn test_kandinsky_localized_threshold() {
        let config = KandinskyConfig {
            bandwidth: 1.0,
            min_effective_samples: 1.0, // Low threshold for test
            ..Default::default()
        };
        let scorer = MockScorer;
        let mut calibrator = KandinskyCalibrator::new(config, scorer);

        // Create two clusters with DIFFERENT score distributions
        let predictions = vec![
            // Cluster 1: High confidence (low scores)
            vec![0.95, 0.05],
            vec![0.90, 0.10],
            // Cluster 2: Low confidence (high scores)
            vec![0.55, 0.45],
            vec![0.60, 0.40],
        ];
        // All correct predictions (label matches argmax)
        let labels = vec![0, 0, 0, 0];
        let features = vec![
            vec![0.0, 0.0],   // Cluster 1 (high confidence)
            vec![0.1, 0.1],
            vec![10.0, 10.0], // Cluster 2 (low confidence)
            vec![10.1, 10.1],
        ];
        let u_values = vec![0.5; 4];

        calibrator.calibrate(&predictions, &labels, &features, &u_values);

        // Query near cluster 1 (high confidence → low threshold)
        let t1 = calibrator.compute_weighted_threshold(&vec![0.0, 0.0], 0.1);

        // Query near cluster 2 (low confidence → high threshold)
        let t2 = calibrator.compute_weighted_threshold(&vec![10.0, 10.0], 0.1);

        // Thresholds should differ: cluster 2 has higher nonconformity scores
        assert!(t2 > t1, "Low confidence cluster should have higher threshold");
    }

    #[test]
    fn test_effective_sample_size() {
        let config = KandinskyConfig {
            bandwidth: 1.0,
            ..Default::default()
        };
        let scorer = MockScorer;
        let mut calibrator = KandinskyCalibrator::new(config, scorer);

        let predictions = vec![vec![0.5, 0.5]; 10];
        let labels = vec![0; 10];
        let features = vec![vec![0.0]; 10]; // All same features
        let u_values = vec![0.5; 10];

        calibrator.calibrate(&predictions, &labels, &features, &u_values);

        // Query at same location should have high effective sample size
        let eff_n = calibrator.effective_sample_size(&vec![0.0]);
        assert!(eff_n > 5.0, "Effective sample size should be substantial");

        // Query far away should have low effective sample size
        let eff_n_far = calibrator.effective_sample_size(&vec![100.0]);
        assert!(eff_n_far < 1.0, "Effective sample size should be low for distant query");
    }
}
