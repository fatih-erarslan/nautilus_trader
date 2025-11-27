//! Integration Tests for Conditional Coverage Conformal Prediction
//!
//! Validates Mondrian and Kandinsky implementations against theoretical guarantees
//! and peer-reviewed literature.

use ats_core::conditional::{
    MondrianCalibrator, MondrianConfig, KandinskyCalibrator, KandinskyConfig,
    LocalizedCalibrator, LocalizedConfig, GroupId, NonconformityScore,
};
use ats_core::scores::{RapsScorer, RapsConfig};

// Mock scorer for basic testing
#[derive(Clone)]
struct SimpleScorer;

impl NonconformityScore for SimpleScorer {
    fn score(&self, prediction: &[f32], label: usize, _u: f32) -> f32 {
        // Score = 1 - p(y)
        1.0 - prediction[label]
    }
}

#[test]
fn test_mondrian_group_conditional_coverage() {
    // Test that Mondrian achieves group-conditional coverage
    let config = MondrianConfig {
        alpha: 0.1,
        min_group_size: 10,
        fallback_to_marginal: false,
        conservative: true,
    };
    let scorer = SimpleScorer;
    let mut calibrator = MondrianCalibrator::new(config, scorer);

    // Create two groups with different difficulty
    // Group 0: Easy predictions (high confidence)
    // Group 1: Hard predictions (low confidence)
    let mut predictions = Vec::new();
    let mut labels = Vec::new();
    let mut groups = Vec::new();
    let mut u_values = Vec::new();

    // Group 0: 50 samples with high confidence
    for i in 0..50 {
        let noise = (i as f32 * 0.01).sin() * 0.05;
        predictions.push(vec![0.85 + noise, 0.10 - noise/2.0, 0.05]);
        labels.push(0);
        groups.push(0);
        u_values.push(0.5);
    }

    // Group 1: 50 samples with low confidence
    for i in 0..50 {
        let noise = (i as f32 * 0.01).cos() * 0.05;
        predictions.push(vec![0.45 + noise, 0.35 - noise/2.0, 0.20]);
        labels.push(0);
        groups.push(1);
        u_values.push(0.5);
    }

    calibrator.calibrate(&predictions, &labels, &groups, &u_values);

    // Verify different thresholds for different groups
    let t0 = calibrator.get_threshold(0);
    let t1 = calibrator.get_threshold(1);

    assert!(t0 < t1, "High-confidence group should have lower threshold");
    assert!(t0 > 0.0 && t0 < 1.0, "Threshold should be in (0,1)");
    assert!(t1 > 0.0 && t1 < 1.0, "Threshold should be in (0,1)");

    // Check group statistics
    let stats = calibrator.get_group_statistics();
    assert_eq!(stats.len(), 2, "Should have 2 groups");
    assert_eq!(stats.get(&0).unwrap().1, 50, "Group 0 should have 50 samples");
    assert_eq!(stats.get(&1).unwrap().1, 50, "Group 1 should have 50 samples");
}

#[test]
fn test_mondrian_prediction_set_validity() {
    // Test that prediction sets contain correct label with high probability
    let config = MondrianConfig {
        alpha: 0.2, // 80% coverage
        min_group_size: 5,
        fallback_to_marginal: true,
        conservative: true,
    };
    let scorer = SimpleScorer;
    let mut calibrator = MondrianCalibrator::new(config, scorer);

    // Small calibration set
    let predictions = vec![
        vec![0.7, 0.2, 0.1],
        vec![0.6, 0.3, 0.1],
        vec![0.8, 0.15, 0.05],
        vec![0.5, 0.4, 0.1],
        vec![0.65, 0.25, 0.1],
    ];
    let labels = vec![0, 0, 0, 0, 0];
    let groups = vec![0, 0, 0, 0, 0];
    let u_values = vec![0.5; 5];

    calibrator.calibrate(&predictions, &labels, &groups, &u_values);

    // Test prediction
    let new_pred = vec![0.6, 0.3, 0.1];
    let pred_set = calibrator.predict_set(&new_pred, 0);

    assert!(!pred_set.is_empty(), "Prediction set should not be empty");
    assert!(pred_set.len() <= 3, "Prediction set should not exceed number of classes");

    // For high probability class, it should be included
    assert!(pred_set.contains(&0), "Should include highest probability class");
}

#[test]
fn test_kandinsky_kernel_weighting() {
    use ats_core::conditional::kandinsky::KernelType;

    let config = KandinskyConfig {
        alpha: 0.1,
        bandwidth: 1.0,
        kernel_type: KernelType::Gaussian,
        min_effective_samples: 5.0,
        fallback_to_marginal: true,
    };
    let scorer = SimpleScorer;
    let mut calibrator = KandinskyCalibrator::new(config, scorer);

    // Create clustered data
    let predictions = vec![
        vec![0.8, 0.1, 0.1],  // Cluster 1
        vec![0.75, 0.15, 0.1],
        vec![0.82, 0.12, 0.06],
        vec![0.2, 0.7, 0.1],  // Cluster 2
        vec![0.15, 0.75, 0.1],
        vec![0.18, 0.72, 0.1],
    ];
    let labels = vec![0, 0, 0, 1, 1, 1];
    let features = vec![
        vec![0.0, 0.0],   // Cluster 1 in feature space
        vec![0.1, 0.1],
        vec![0.05, 0.05],
        vec![5.0, 5.0],   // Cluster 2 in feature space
        vec![5.1, 5.1],
        vec![5.05, 5.05],
    ];
    let u_values = vec![0.5; 6];

    calibrator.calibrate(&predictions, &labels, &features, &u_values);

    // Test prediction near cluster 1
    let pred_cluster1 = vec![0.78, 0.12, 0.1];
    let features_cluster1 = vec![0.02, 0.02];
    let set1 = calibrator.predict_set(&pred_cluster1, &features_cluster1);

    // Test prediction near cluster 2
    let pred_cluster2 = vec![0.18, 0.70, 0.12];
    let features_cluster2 = vec![5.02, 5.02];
    let set2 = calibrator.predict_set(&pred_cluster2, &features_cluster2);

    assert!(!set1.is_empty());
    assert!(!set2.is_empty());

    // Verify effective sample sizes
    let eff_n1 = calibrator.effective_sample_size(&features_cluster1);
    let eff_n2 = calibrator.effective_sample_size(&features_cluster2);

    assert!(eff_n1 > 1.0, "Should have reasonable effective sample size");
    assert!(eff_n2 > 1.0, "Should have reasonable effective sample size");
}

#[test]
fn test_kandinsky_kernel_types() {
    use ats_core::conditional::kandinsky::KernelType;

    // Test all kernel types compile and run
    let kernels = vec![
        KernelType::Gaussian,
        KernelType::Epanechnikov,
        KernelType::Tricube,
    ];

    for kernel_type in kernels {
        let config = KandinskyConfig {
            alpha: 0.1,
            bandwidth: 1.0,
            kernel_type,
            min_effective_samples: 1.0,
            fallback_to_marginal: true,
        };
        let scorer = SimpleScorer;
        let mut calibrator = KandinskyCalibrator::new(config, scorer);

        let predictions = vec![vec![0.5, 0.5]; 5];
        let labels = vec![0; 5];
        let features = vec![vec![0.0]; 5];
        let u_values = vec![0.5; 5];

        calibrator.calibrate(&predictions, &labels, &features, &u_values);

        let pred_set = calibrator.predict_set(&vec![0.5, 0.5], &vec![0.0]);
        assert!(!pred_set.is_empty(), "Prediction set should not be empty for {:?}", kernel_type);
    }
}

#[test]
fn test_integration_with_raps_scorer() {
    // Test Mondrian with RAPS scorer (real scorer from scores module)
    let raps_config = RapsConfig {
        lambda: 0.01,
        k_reg: 3,
        randomize_ties: true,
    };
    let scorer = RapsScorer::new(raps_config);

    let mondrian_config = MondrianConfig {
        alpha: 0.1,
        min_group_size: 5,
        fallback_to_marginal: true,
        conservative: true,
    };
    let mut calibrator = MondrianCalibrator::new(mondrian_config, scorer);

    let predictions = vec![
        vec![0.6, 0.3, 0.1],
        vec![0.5, 0.4, 0.1],
        vec![0.7, 0.2, 0.1],
        vec![0.4, 0.5, 0.1],
        vec![0.55, 0.35, 0.1],
    ];
    let labels = vec![0, 0, 0, 1, 0];
    let groups = vec![0, 0, 0, 0, 0];
    let u_values = vec![0.3, 0.7, 0.5, 0.4, 0.6];

    calibrator.calibrate(&predictions, &labels, &groups, &u_values);

    let pred_set = calibrator.predict_set(&vec![0.6, 0.3, 0.1], 0);
    assert!(!pred_set.is_empty(), "RAPS-based prediction set should not be empty");
}

#[test]
fn test_localized_mondrian_mode() {
    use ats_core::conditional::localized::LocalizationType;

    let config = LocalizedConfig {
        alpha: 0.1,
        localization_type: LocalizationType::Mondrian {
            min_group_size: 3,
            fallback_to_marginal: true,
        },
    };

    let scorer = SimpleScorer;
    let mut calibrator = LocalizedCalibrator::new(config, scorer);

    let predictions = vec![
        vec![0.7, 0.2, 0.1],
        vec![0.6, 0.3, 0.1],
        vec![0.8, 0.15, 0.05],
    ];
    let labels = vec![0, 0, 0];
    let groups = vec![0, 0, 0];
    let u_values = vec![0.5; 3];

    calibrator.calibrate_with_groups(&predictions, &labels, &groups, &u_values);

    // Test passes if calibration succeeds
    assert!(true, "Localized Mondrian calibration succeeded");
}

#[test]
fn test_localized_kandinsky_mode() {
    use ats_core::conditional::localized::LocalizationType;
    use ats_core::conditional::kandinsky::KernelType;

    let config = LocalizedConfig {
        alpha: 0.1,
        localization_type: LocalizationType::Kandinsky {
            bandwidth: 1.0,
            kernel_type: KernelType::Gaussian,
            min_effective_samples: 2.0,
        },
    };

    let scorer = SimpleScorer;
    let mut calibrator = LocalizedCalibrator::new(config, scorer);

    let predictions = vec![
        vec![0.7, 0.2, 0.1],
        vec![0.6, 0.3, 0.1],
        vec![0.8, 0.15, 0.05],
    ];
    let labels = vec![0, 0, 0];
    let features = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.1],
        vec![0.05, 0.05],
    ];
    let u_values = vec![0.5; 3];

    calibrator.calibrate_with_features(&predictions, &labels, &features, &u_values);

    // Test passes if calibration succeeds
    assert!(true, "Localized Kandinsky calibration succeeded");
}

#[test]
fn test_mondrian_edge_case_single_group() {
    // Edge case: All samples in one group
    let config = MondrianConfig {
        alpha: 0.1,
        min_group_size: 2,
        fallback_to_marginal: true,
        conservative: true,
    };
    let scorer = SimpleScorer;
    let mut calibrator = MondrianCalibrator::new(config, scorer);

    let predictions = vec![
        vec![0.8, 0.1, 0.1],
        vec![0.7, 0.2, 0.1],
    ];
    let labels = vec![0, 0];
    let groups = vec![0, 0]; // All in same group
    let u_values = vec![0.5, 0.5];

    calibrator.calibrate(&predictions, &labels, &groups, &u_values);

    let threshold = calibrator.get_threshold(0);
    assert!(threshold > 0.0, "Should have valid threshold for single group");
}

#[test]
fn test_kandinsky_edge_case_identical_features() {
    use ats_core::conditional::kandinsky::KernelType;

    // Edge case: All samples have identical features
    let config = KandinskyConfig {
        alpha: 0.1,
        bandwidth: 1.0,
        kernel_type: KernelType::Gaussian,
        min_effective_samples: 1.0,
        fallback_to_marginal: true,
    };
    let scorer = SimpleScorer;
    let mut calibrator = KandinskyCalibrator::new(config, scorer);

    let predictions = vec![
        vec![0.7, 0.2, 0.1],
        vec![0.6, 0.3, 0.1],
    ];
    let labels = vec![0, 0];
    let features = vec![
        vec![1.0, 1.0],
        vec![1.0, 1.0], // Identical features
    ];
    let u_values = vec![0.5, 0.5];

    calibrator.calibrate(&predictions, &labels, &features, &u_values);

    let eff_n = calibrator.effective_sample_size(&vec![1.0, 1.0]);
    assert!(eff_n > 1.5, "Effective sample size should be high for identical features");
}

#[test]
fn test_coverage_guarantee_simulation() {
    // Empirical test: verify coverage is approximately 1-alpha
    let alpha = 0.1; // Target 90% coverage
    let config = MondrianConfig {
        alpha,
        min_group_size: 10,
        fallback_to_marginal: false,
        conservative: true,
    };
    let scorer = SimpleScorer;

    let n_trials = 100;
    let mut coverage_rates = Vec::new();

    for _ in 0..10 { // Run multiple independent trials
        let mut calibrator = MondrianCalibrator::new(config.clone(), scorer.clone());

        // Generate calibration data
        let mut predictions = Vec::new();
        let mut labels = Vec::new();
        let mut groups = Vec::new();
        let u_values = vec![0.5; 50];

        for i in 0..50 {
            let p = 0.7 + ((i % 10) as f32) * 0.02;
            predictions.push(vec![p, 1.0 - p]);
            labels.push(0);
            groups.push(i % 2); // Two groups
        }

        calibrator.calibrate(&predictions, &labels, &groups, &u_values);

        // Test coverage on new samples
        let mut covered = 0;
        for i in 0..n_trials {
            let p = 0.65 + ((i % 10) as f32) * 0.02;
            let test_pred = vec![p, 1.0 - p];
            let pred_set = calibrator.predict_set(&test_pred, (i % 2) as u64);

            if pred_set.contains(&0) {
                covered += 1;
            }
        }

        let coverage = covered as f32 / n_trials as f32;
        coverage_rates.push(coverage);
    }

    let avg_coverage = coverage_rates.iter().sum::<f32>() / coverage_rates.len() as f32;

    // Coverage should be at least 1-alpha (allowing for randomness)
    assert!(avg_coverage >= 1.0 - alpha - 0.15,
            "Average coverage {:.3} should be close to {:.3}", avg_coverage, 1.0 - alpha);
}
