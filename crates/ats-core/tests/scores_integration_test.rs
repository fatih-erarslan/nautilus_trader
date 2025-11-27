//! Integration tests for nonconformity scores module
//!
//! Tests all scorers comprehensively to ensure:
//! 1. Mathematical correctness according to peer-reviewed literature
//! 2. Performance targets (<3μs per sample for RAPS)
//! 3. Numerical stability
//! 4. Batch processing efficiency

use ats_core::scores::*;
use std::time::Instant;

/// Test helper: Generate uniform random values
fn generate_u_values(n: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen::<f32>()).collect()
}

/// Test helper: Generate realistic softmax probabilities
fn generate_softmax(n_classes: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut probs: Vec<f32> = (0..n_classes)
        .map(|_| rng.gen::<f32>())
        .collect();

    let sum: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= sum);

    probs
}

#[test]
fn test_raps_mathematical_correctness() {
    // Test against manually computed example from Romano et al. (2020)
    let config = RapsConfig {
        lambda: 0.01,
        k_reg: 5,
        randomize_ties: true,
    };
    let scorer = RapsScorer::new(config);

    let softmax = vec![0.6, 0.3, 0.1];
    let true_label = 1;
    let u = 0.5;

    let score = scorer.score(&softmax, true_label, u);

    // Expected computation:
    // - Classes sorted by prob: [0.6, 0.3, 0.1]
    // - True label (0.3) has rank 2 (0-indexed: 1)
    // - cumsum before true = 0.6
    // - base_score = 0.6 + 0.5 * 0.3 = 0.75
    // - reg_term = 0 (rank 2 < k_reg 5)
    // - total = 0.75

    assert!((score - 0.75).abs() < 1e-6,
        "RAPS score mismatch: expected 0.75, got {}", score);
}

#[test]
fn test_raps_with_regularization() {
    let config = RapsConfig {
        lambda: 0.1,
        k_reg: 2,
        randomize_ties: false,
    };
    let scorer = RapsScorer::new(config);

    // Create case where regularization kicks in
    let softmax = vec![0.4, 0.25, 0.15, 0.12, 0.08];
    let true_label = 4; // Ranked 5th
    let u = 0.5;

    let score = scorer.score(&softmax, true_label, u);

    // cumsum = 0.4 + 0.25 + 0.15 + 0.12 = 0.92
    // base = 0.92 + 0.5 * 0.08 = 0.96
    // reg = 0.1 * (5 - 2) = 0.3
    // total = 1.26

    assert!((score - 1.26).abs() < 1e-6,
        "RAPS with regularization: expected 1.26, got {}", score);
}

#[test]
fn test_aps_equals_raps_with_zero_lambda() {
    let aps_scorer = ApsScorer::default();
    let raps_scorer = RapsScorer::new(RapsConfig {
        lambda: 0.0,
        k_reg: 0,
        randomize_ties: true,
    });

    let softmax = vec![0.6, 0.3, 0.1];
    let true_label = 1;
    let u = 0.5;

    let aps_score = aps_scorer.score(&softmax, true_label, u);
    let raps_score = raps_scorer.score(&softmax, true_label, u);

    assert!((aps_score - raps_score).abs() < 1e-6,
        "APS should equal RAPS with λ=0: APS={}, RAPS={}", aps_score, raps_score);
}

#[test]
fn test_threshold_score_bounds() {
    let scorer = ThresholdScorer::default();

    // High confidence case
    let high_conf = vec![0.95, 0.03, 0.02];
    let score_high = scorer.score(&high_conf, 0, 0.0);
    assert!((score_high - 0.05).abs() < 1e-6,
        "High confidence should give low score");

    // Low confidence case
    let low_conf = vec![0.34, 0.33, 0.33];
    let score_low = scorer.score(&low_conf, 0, 0.0);
    assert!((score_low - 0.66).abs() < 0.01,
        "Low confidence should give high score");

    // Score should be in [0, 1]
    assert!(score_high >= 0.0 && score_high <= 1.0);
    assert!(score_low >= 0.0 && score_low <= 1.0);
}

#[test]
fn test_lac_weight_effects() {
    // Uniform weights
    let uniform_weights = vec![1.0, 1.0, 1.0];
    let scorer_uniform = LacScorer::with_weights(uniform_weights);

    // Weighted (emphasize first class)
    let custom_weights = vec![10.0, 1.0, 1.0];
    let scorer_custom = LacScorer::with_weights(custom_weights);

    let softmax = vec![0.6, 0.3, 0.1];
    let true_label = 1;
    let u = 0.5;

    let score_uniform = scorer_uniform.score(&softmax, true_label, u);
    let score_custom = scorer_custom.score(&softmax, true_label, u);

    assert!(score_custom > score_uniform,
        "Higher weight on non-true class should increase score");
}

#[test]
fn test_saps_size_penalty() {
    let no_penalty = SapsScorer::new(SapsConfig {
        size_penalty: 0.0,
        randomize_ties: true,
    });

    let with_penalty = SapsScorer::new(SapsConfig {
        size_penalty: 0.1,
        randomize_ties: true,
    });

    let softmax = vec![0.6, 0.3, 0.1];
    let true_label = 1;
    let u = 0.5;

    let score_no_penalty = no_penalty.score(&softmax, true_label, u);
    let score_with_penalty = with_penalty.score(&softmax, true_label, u);

    assert!(score_with_penalty > score_no_penalty,
        "Size penalty should increase score");
}

#[test]
fn test_raps_performance_target() {
    // Target: <3μs per sample
    let scorer = RapsScorer::default();
    let n_samples = 10_000;

    let u_values = generate_u_values(n_samples);
    let mut total_time = std::time::Duration::ZERO;

    for i in 0..n_samples {
        let softmax = generate_softmax(100); // 100 classes
        let true_label = i % 100;
        let u = u_values[i];

        let start = Instant::now();
        let _ = scorer.score(&softmax, true_label, u);
        total_time += start.elapsed();
    }

    let avg_time_us = total_time.as_micros() as f64 / n_samples as f64;

    println!("RAPS average time per sample: {:.3}μs", avg_time_us);
    assert!(avg_time_us < 3.0,
        "RAPS failed performance target: {:.3}μs per sample (target: <3μs)",
        avg_time_us);
}

#[test]
fn test_batch_processing_efficiency() {
    let scorer = RapsScorer::default();
    let n_samples = 1000;
    let n_classes = 50;

    // Generate batch data
    let batch: Vec<Vec<f32>> = (0..n_samples)
        .map(|_| generate_softmax(n_classes))
        .collect();

    let labels: Vec<usize> = (0..n_samples)
        .map(|i| i % n_classes)
        .collect();

    let u_values = generate_u_values(n_samples);

    // Time batch processing
    let start = Instant::now();
    let scores = scorer.score_batch(&batch, &labels, &u_values);
    let batch_time = start.elapsed();

    assert_eq!(scores.len(), n_samples);

    let avg_time_us = batch_time.as_micros() as f64 / n_samples as f64;
    println!("Batch processing average: {:.3}μs per sample", avg_time_us);

    // Batch should be faster than 5μs per sample
    assert!(avg_time_us < 5.0,
        "Batch processing too slow: {:.3}μs per sample", avg_time_us);
}

#[test]
fn test_numerical_stability() {
    let scorer = RapsScorer::default();

    // Test with very small probabilities
    let tiny_probs = vec![1e-10, 1e-10, 1.0 - 2e-10];
    let score1 = scorer.score(&tiny_probs, 2, 0.5);
    assert!(score1.is_finite(), "Score should be finite with tiny probs");

    // Test with nearly equal probabilities
    let equal_probs = vec![0.3333, 0.3333, 0.3334];
    let score2 = scorer.score(&equal_probs, 0, 0.5);
    assert!(score2.is_finite(), "Score should be finite with equal probs");

    // Test with extreme probabilities
    let extreme = vec![0.99, 0.005, 0.005];
    let score3 = scorer.score(&extreme, 0, 0.5);
    assert!(score3.is_finite(), "Score should be finite with extreme probs");
}

#[test]
fn test_monotonicity_property() {
    // Score should increase as true label probability decreases
    let scorer = ApsScorer::default();

    let high_prob = vec![0.1, 0.8, 0.1]; // True label has 0.8
    let low_prob = vec![0.4, 0.2, 0.4];  // True label has 0.2

    let score_high = scorer.score(&high_prob, 1, 0.5);
    let score_low = scorer.score(&low_prob, 1, 0.5);

    assert!(score_low > score_high,
        "Lower probability should yield higher nonconformity score");
}

#[test]
fn test_u_value_effect() {
    // Score should increase monotonically with u
    let scorer = RapsScorer::default();
    let softmax = vec![0.6, 0.3, 0.1];
    let true_label = 1;

    let score_u0 = scorer.score(&softmax, true_label, 0.0);
    let score_u05 = scorer.score(&softmax, true_label, 0.5);
    let score_u1 = scorer.score(&softmax, true_label, 1.0);

    assert!(score_u1 > score_u05 && score_u05 > score_u0,
        "Score should increase monotonically with u");
}

#[test]
fn test_all_scorers_consistency() {
    // All scorers should produce valid scores for the same input
    let softmax = vec![0.6, 0.3, 0.1];
    let true_label = 1;
    let u = 0.5;

    let raps = RapsScorer::default().score(&softmax, true_label, u);
    let aps = ApsScorer::default().score(&softmax, true_label, u);
    let saps = SapsScorer::default().score(&softmax, true_label, u);
    let thr = ThresholdScorer::default().score(&softmax, true_label, u);
    let lac = LacScorer::default().score(&softmax, true_label, u);

    assert!(raps.is_finite() && raps >= 0.0, "RAPS score invalid");
    assert!(aps.is_finite() && aps >= 0.0, "APS score invalid");
    assert!(saps.is_finite() && saps >= 0.0, "SAPS score invalid");
    assert!(thr.is_finite() && thr >= 0.0 && thr <= 1.0, "THR score invalid");
    assert!(lac.is_finite() && lac >= 0.0, "LAC score invalid");

    println!("RAPS: {:.4}, APS: {:.4}, SAPS: {:.4}, THR: {:.4}, LAC: {:.4}",
        raps, aps, saps, thr, lac);
}

#[test]
fn test_large_scale_batch() {
    // Test with realistic large-scale data
    let scorer = RapsScorer::default();
    let n_samples = 10_000;
    let n_classes = 1000;

    let batch: Vec<Vec<f32>> = (0..n_samples)
        .map(|_| generate_softmax(n_classes))
        .collect();

    let labels: Vec<usize> = (0..n_samples)
        .map(|i| i % n_classes)
        .collect();

    let u_values = generate_u_values(n_samples);

    let start = Instant::now();
    let scores = scorer.score_batch(&batch, &labels, &u_values);
    let duration = start.elapsed();

    assert_eq!(scores.len(), n_samples);
    assert!(scores.iter().all(|&s| s.is_finite()));

    println!("Large-scale batch ({} samples, {} classes): {:.2}ms total, {:.3}μs/sample",
        n_samples, n_classes,
        duration.as_millis(),
        duration.as_micros() as f64 / n_samples as f64);
}
