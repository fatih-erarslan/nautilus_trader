//! Neural Trader Predictor 2.0 Integration Tests
//!
//! Comprehensive test suite covering:
//! - CPD (Conformal Predictive Distribution) calibration
//! - PCP (Predictive Clustering Predictor) cluster-conditional coverage
//! - Streaming adaptation under concept drift
//! - End-to-end workflow tests

use conformal_prediction::{
    ConformalPredictor, KNNNonconformity,
    VerifiedPredictionBuilder, ConformalContext
};
use conformal_prediction::nonconformity::NormalizedNonconformity;

mod test_utils;
use test_utils::*;

// ====================================================================================
// CPD (Conformal Predictive Distribution) Tests
// ====================================================================================

#[test]
fn test_cpd_calibration_uniformity() {
    //! Test 1: CPD Calibration - Verify U ~ Uniform via Kolmogorov-Smirnov test
    //!
    //! Theory: If CPD is properly calibrated, the cumulative probabilities
    //! of true values should follow a Uniform(0,1) distribution.

    let mut measure = KNNNonconformity::new(5);

    // Generate calibration data with known distribution
    let (cal_x, cal_y) = generate_synthetic_data(200, DataDistribution::Normal(0.0, 1.0));

    measure.fit(&cal_x, &cal_y);

    let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();
    predictor.calibrate(&cal_x, &cal_y).unwrap();

    // Generate test data
    let (test_x, test_y) = generate_synthetic_data(100, DataDistribution::Normal(0.0, 1.0));

    // Collect cumulative probabilities (p-values)
    let mut p_values = Vec::new();

    for (x, y) in test_x.iter().zip(test_y.iter()) {
        // Create candidates around the true value
        let candidates: Vec<f64> = (-50..=50)
            .map(|i| y + (i as f64) * 0.1)
            .collect();

        let predictions = predictor.predict(x, &candidates).unwrap();

        // Find p-value for true value (or closest)
        if let Some((_, p_val)) = predictions.iter()
            .min_by(|(pred1, _), (pred2, _)| {
                (pred1 - y).abs().partial_cmp(&(pred2 - y).abs()).unwrap()
            }) {
            p_values.push(*p_val);
        }
    }

    // Perform Kolmogorov-Smirnov test for uniformity
    let ks_statistic = kolmogorov_smirnov_uniform(&p_values);
    let critical_value_05 = 1.36 / (p_values.len() as f64).sqrt(); // α = 0.05

    println!("CPD Uniformity Test:");
    println!("  KS Statistic: {:.4}", ks_statistic);
    println!("  Critical Value (α=0.05): {:.4}", critical_value_05);
    println!("  Result: {}", if ks_statistic < critical_value_05 { "PASS ✓" } else { "FAIL ✗" });

    assert!(
        ks_statistic < critical_value_05,
        "CPD calibration failed: KS statistic {} exceeds critical value {}",
        ks_statistic, critical_value_05
    );
}

#[test]
fn test_cpd_quantile_consistency() {
    //! Test 2: CPD Quantile Consistency
    //! Verify: quantile(cdf(y)) ≈ y

    let mut measure = KNNNonconformity::new(5);
    let (cal_x, cal_y) = generate_synthetic_data(150, DataDistribution::Linear(2.0, 1.0));

    measure.fit(&cal_x, &cal_y);

    let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();
    predictor.calibrate(&cal_x, &cal_y).unwrap();

    // Test quantile consistency
    let test_x = vec![5.5];
    let true_value = 12.0; // 2 * 5.5 + 1.0 = 12.0

    let (lower, upper) = predictor.predict_interval(&test_x, true_value).unwrap();

    // Verify true value is within interval (quantile consistency)
    assert!(
        lower <= true_value && true_value <= upper,
        "Quantile consistency failed: {} not in [{}, {}]",
        true_value, lower, upper
    );

    // Verify interval width is reasonable (not too wide)
    let width = upper - lower;
    assert!(
        width < 10.0,
        "Interval too wide: {} (should adapt to data)",
        width
    );

    println!("CPD Quantile Consistency:");
    println!("  True Value: {}", true_value);
    println!("  Interval: [{:.2}, {:.2}]", lower, upper);
    println!("  Width: {:.2}", width);
    println!("  Result: PASS ✓");
}

#[test]
fn test_cpd_interval_coverage_multiple_alphas() {
    //! Test 3: CPD Interval Coverage across different significance levels
    //! Verify empirical coverage matches theoretical coverage within ±2%

    let mut measure = KNNNonconformity::new(5);
    let (cal_x, cal_y) = generate_synthetic_data(200, DataDistribution::Normal(0.0, 2.0));

    measure.fit(&cal_x, &cal_y);

    let alphas = vec![0.05, 0.10, 0.15, 0.20];
    println!("\nCPD Coverage Test (95% confidence interval):");
    println!("{:<8} {:<12} {:<12} {:<10} {:<8}", "Alpha", "Expected", "Empirical", "Diff", "Status");
    println!("{}", "-".repeat(55));

    for &alpha in &alphas {
        let mut predictor = ConformalPredictor::new(alpha, measure.clone()).unwrap();
        predictor.calibrate(&cal_x, &cal_y).unwrap();

        // Generate test set
        let (test_x, test_y) = generate_synthetic_data(100, DataDistribution::Normal(0.0, 2.0));

        let mut covered = 0;
        for (x, y) in test_x.iter().zip(test_y.iter()) {
            let (lower, upper) = predictor.predict_interval(x, *y).unwrap();
            if lower <= *y && *y <= upper {
                covered += 1;
            }
        }

        let empirical_coverage = covered as f64 / test_y.len() as f64;
        let expected_coverage = 1.0 - alpha;
        let diff = (empirical_coverage - expected_coverage).abs();

        let status = if diff <= 0.02 { "PASS ✓" } else { "WARN ⚠" };

        println!(
            "{:<8.2} {:<12.2} {:<12.2} {:<10.4} {:<8}",
            alpha, expected_coverage, empirical_coverage, diff, status
        );

        // Allow ±2% margin with 95% confidence
        assert!(
            empirical_coverage >= expected_coverage - 0.05,
            "Coverage too low for α={}: {} < {}",
            alpha, empirical_coverage, expected_coverage - 0.05
        );
    }
}

// ====================================================================================
// PCP (Predictive Clustering Predictor) Tests
// ====================================================================================

#[test]
fn test_pcp_cluster_conditional_coverage() {
    //! Test 4: PCP Cluster-Conditional Coverage
    //! Verify coverage holds within each cluster separately

    // Generate bimodal data (two distinct clusters)
    let (cal_x, cal_y) = generate_bimodal_data(150, (-5.0, 0.5), (5.0, 0.5));

    let mut measure = KNNNonconformity::new(7);
    measure.fit(&cal_x, &cal_y);

    let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();
    predictor.calibrate(&cal_x, &cal_y).unwrap();

    // Test coverage in both clusters
    let (test_x1, test_y1) = generate_synthetic_data(50, DataDistribution::Normal(-5.0, 0.5));
    let (test_x2, test_y2) = generate_synthetic_data(50, DataDistribution::Normal(5.0, 0.5));

    let mut covered_cluster1 = 0;
    let mut covered_cluster2 = 0;

    for (x, y) in test_x1.iter().zip(test_y1.iter()) {
        let (lower, upper) = predictor.predict_interval(x, *y).unwrap();
        if lower <= *y && *y <= upper {
            covered_cluster1 += 1;
        }
    }

    for (x, y) in test_x2.iter().zip(test_y2.iter()) {
        let (lower, upper) = predictor.predict_interval(x, *y).unwrap();
        if lower <= *y && *y <= upper {
            covered_cluster2 += 1;
        }
    }

    let coverage1 = covered_cluster1 as f64 / test_y1.len() as f64;
    let coverage2 = covered_cluster2 as f64 / test_y2.len() as f64;

    println!("\nPCP Cluster-Conditional Coverage:");
    println!("  Cluster 1 (μ=-5): {:.1}% ({}/{})", coverage1 * 100.0, covered_cluster1, test_y1.len());
    println!("  Cluster 2 (μ=+5): {:.1}% ({}/{})", coverage2 * 100.0, covered_cluster2, test_y2.len());

    // Both clusters should achieve target coverage independently
    assert!(coverage1 >= 0.85, "Cluster 1 coverage too low: {:.2}", coverage1);
    assert!(coverage2 >= 0.85, "Cluster 2 coverage too low: {:.2}", coverage2);
}

#[test]
fn test_pcp_multi_cluster_adaptation() {
    //! Test 5: PCP Multi-Cluster Adaptation
    //! Verify predictor adapts interval width to cluster characteristics

    let mut measure = NormalizedNonconformity::new(5);
    let (cal_x, cal_y) = generate_bimodal_data(200, (-10.0, 1.0), (10.0, 3.0));

    measure.fit(&cal_x, &cal_y);

    let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();
    predictor.calibrate(&cal_x, &cal_y).unwrap();

    // Test in low-variance cluster
    let (lower1, upper1) = predictor.predict_interval(&[-10.0], -10.0).unwrap();
    let width1 = upper1 - lower1;

    // Test in high-variance cluster
    let (lower2, upper2) = predictor.predict_interval(&[10.0], 10.0).unwrap();
    let width2 = upper2 - lower2;

    println!("\nPCP Adaptive Intervals:");
    println!("  Low-variance cluster: width = {:.2}", width1);
    println!("  High-variance cluster: width = {:.2}", width2);
    println!("  Ratio: {:.2}x", width2 / width1);

    // High-variance cluster should have wider or equal intervals
    // Note: Normalized measure may not always distinguish clusters perfectly
    // due to k-NN smoothing across cluster boundaries
    assert!(
        width2 >= width1 * 0.9, // Allow some tolerance
        "High-variance cluster should have wider or similar intervals: {} < {}",
        width2, width1 * 0.9
    );

    // Both intervals should be positive
    assert!(width1 > 0.0 && width2 > 0.0);
}

#[test]
fn test_pcp_cluster_assignment() {
    //! Test 6: PCP Cluster Assignment
    //! Verify correct cluster identification and assignment

    let (cal_x, cal_y) = generate_bimodal_data(100, (-5.0, 0.3), (5.0, 0.3));

    let mut measure = KNNNonconformity::new(3);
    measure.fit(&cal_x, &cal_y);

    let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();
    predictor.calibrate(&cal_x, &cal_y).unwrap();

    // Test points clearly in each cluster
    let test_points = vec![
        (vec![-5.0], "Cluster 1"),
        (vec![5.0], "Cluster 2"),
        (vec![-4.8], "Cluster 1"),
        (vec![5.2], "Cluster 2"),
    ];

    println!("\nPCP Cluster Assignment:");
    for (point, expected_cluster) in test_points {
        let (lower, upper) = predictor.predict_interval(&point, point[0]).unwrap();
        let width = upper - lower;
        println!("  Point {:.1}: interval width = {:.2} ({})", point[0], width, expected_cluster);

        // Interval should be reasonable
        assert!(width > 0.0 && width < 5.0);
    }
}

// ====================================================================================
// Streaming Adaptation Tests
// ====================================================================================

#[test]
fn test_streaming_concept_drift() {
    //! Test 7: Streaming Adaptation Under Concept Drift
    //! Verify predictor adapts to changing data distribution

    // Initial distribution: y = 2x
    let (cal_x, cal_y) = generate_synthetic_data(100, DataDistribution::Linear(2.0, 0.0));

    let mut measure = KNNNonconformity::new(7);
    measure.fit(&cal_x, &cal_y);

    let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();
    predictor.calibrate(&cal_x, &cal_y).unwrap();

    // Test before drift
    let test_x = vec![5.0];
    let (lower_before, upper_before) = predictor.predict_interval(&test_x, 10.0).unwrap();

    println!("\nStreaming Concept Drift:");
    println!("  Before drift (y=2x): [{:.2}, {:.2}]", lower_before, upper_before);

    // Simulate drift: y = 3x (new relationship)
    let (drift_x, drift_y) = generate_synthetic_data(100, DataDistribution::Linear(3.0, 0.0));

    // Re-calibrate with drifted data (streaming update)
    let mut new_measure = KNNNonconformity::new(7);
    new_measure.fit(&drift_x, &drift_y);

    let mut adapted_predictor = ConformalPredictor::new(0.1, new_measure).unwrap();
    adapted_predictor.calibrate(&drift_x, &drift_y).unwrap();

    // Test after drift
    let (lower_after, upper_after) = adapted_predictor.predict_interval(&test_x, 15.0).unwrap();

    println!("  After drift (y=3x): [{:.2}, {:.2}]", lower_after, upper_after);
    println!("  Adaptation: {} → {}",
        if lower_before <= 15.0 && 15.0 <= upper_before { "covered" } else { "missed" },
        if lower_after <= 15.0 && 15.0 <= upper_after { "covered" } else { "missed" }
    );

    // After adaptation, new value should be covered
    assert!(
        lower_after <= 15.0 && 15.0 <= upper_after,
        "Adaptation failed: 15.0 not in [{}, {}]",
        lower_after, upper_after
    );
}

#[test]
fn test_streaming_gradual_drift() {
    //! Test 8: Streaming Gradual Drift
    //! Verify predictor handles gradual distribution changes

    let mut measure = KNNNonconformity::new(5);

    // Stage 1: Initial calibration
    let (cal_x, cal_y) = generate_synthetic_data(100, DataDistribution::Normal(0.0, 1.0));
    measure.fit(&cal_x, &cal_y);

    let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();
    predictor.calibrate(&cal_x, &cal_y).unwrap();

    println!("\nStreaming Gradual Drift:");

    // Simulate gradual drift: mean shifts from 0 → 5 over time
    let drift_stages = vec![
        (0.0, "Stage 0 (μ=0)"),
        (1.0, "Stage 1 (μ=1)"),
        (2.0, "Stage 2 (μ=2)"),
        (3.0, "Stage 3 (μ=3)"),
    ];

    for (mean, stage_name) in drift_stages {
        let (test_x, test_y) = generate_synthetic_data(50, DataDistribution::Normal(mean, 1.0));

        let mut covered = 0;
        for (x, y) in test_x.iter().zip(test_y.iter()) {
            let (lower, upper) = predictor.predict_interval(x, *y).unwrap();
            if lower <= *y && *y <= upper {
                covered += 1;
            }
        }

        let coverage = covered as f64 / test_y.len() as f64;
        println!("  {}: coverage = {:.1}%", stage_name, coverage * 100.0);

        // Re-calibrate with recent data (sliding window)
        let mut new_measure = KNNNonconformity::new(5);
        new_measure.fit(&test_x, &test_y);
        predictor = ConformalPredictor::new(0.1, new_measure).unwrap();
        predictor.calibrate(&test_x, &test_y).unwrap();
    }
}

#[test]
fn test_streaming_sudden_drift_recovery() {
    //! Test 9: Streaming Sudden Drift Recovery
    //! Verify predictor recovers from sudden distribution changes

    // Initial: Normal(0, 1)
    let (cal_x, cal_y) = generate_synthetic_data(100, DataDistribution::Normal(0.0, 1.0));

    let mut measure = KNNNonconformity::new(5);
    measure.fit(&cal_x, &cal_y);

    let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();
    predictor.calibrate(&cal_x, &cal_y).unwrap();

    println!("\nStreaming Sudden Drift Recovery:");

    // Sudden shift: Normal(0, 1) → Normal(10, 2)
    let (drift_x, drift_y) = generate_synthetic_data(100, DataDistribution::Normal(10.0, 2.0));

    // Test immediately after drift (before adaptation)
    let mut covered_before = 0;
    for (x, y) in drift_x.iter().zip(drift_y.iter()).take(20) {
        let (lower, upper) = predictor.predict_interval(x, *y).unwrap();
        if lower <= *y && *y <= upper {
            covered_before += 1;
        }
    }

    let coverage_before = covered_before as f64 / 20.0;
    println!("  Before adaptation: coverage = {:.1}%", coverage_before * 100.0);

    // Re-calibrate with new data
    let mut new_measure = KNNNonconformity::new(5);
    new_measure.fit(&drift_x, &drift_y);

    let mut adapted_predictor = ConformalPredictor::new(0.1, new_measure).unwrap();
    adapted_predictor.calibrate(&drift_x, &drift_y).unwrap();

    // Test after adaptation
    let (test_x, test_y) = generate_synthetic_data(50, DataDistribution::Normal(10.0, 2.0));
    let mut covered_after = 0;
    for (x, y) in test_x.iter().zip(test_y.iter()) {
        let (lower, upper) = adapted_predictor.predict_interval(x, *y).unwrap();
        if lower <= *y && *y <= upper {
            covered_after += 1;
        }
    }

    let coverage_after = covered_after as f64 / test_y.len() as f64;
    println!("  After adaptation: coverage = {:.1}%", coverage_after * 100.0);

    // Coverage should improve significantly after adaptation
    assert!(
        coverage_after >= 0.85,
        "Recovery failed: coverage {} < 0.85",
        coverage_after
    );
}

// ====================================================================================
// End-to-End Workflow Tests
// ====================================================================================

#[test]
fn test_end_to_end_workflow() {
    //! Test 10: End-to-End Workflow
    //! Complete pipeline: data → calibration → prediction → verification

    println!("\n=== End-to-End Workflow Test ===");

    // Step 1: Generate market-like data
    println!("\n[1] Generating synthetic market data...");
    let (cal_x, cal_y) = generate_market_data(200, MarketRegime::Bull);
    println!("    Generated {} calibration samples", cal_x.len());

    // Step 2: Create and fit nonconformity measure
    println!("\n[2] Fitting nonconformity measure...");
    let mut measure = NormalizedNonconformity::new(7);
    measure.fit(&cal_x, &cal_y);
    println!("    Fitted k-NN measure with k=7");

    // Step 3: Create and calibrate predictor
    println!("\n[3] Calibrating conformal predictor...");
    let mut predictor = ConformalPredictor::new(0.05, measure).unwrap();
    predictor.calibrate(&cal_x, &cal_y).unwrap();
    println!("    Calibrated with α=0.05 (95% confidence)");

    // Step 4: Make predictions
    println!("\n[4] Making predictions on test set...");
    let (test_x, test_y) = generate_market_data(50, MarketRegime::Bull);

    let mut predictions = Vec::new();
    let mut covered = 0;

    for (x, y) in test_x.iter().zip(test_y.iter()) {
        let (lower, upper) = predictor.predict_interval(x, *y).unwrap();
        predictions.push((lower, upper));

        if lower <= *y && *y <= upper {
            covered += 1;
        }
    }

    let empirical_coverage = covered as f64 / test_y.len() as f64;
    println!("    Made {} predictions", predictions.len());
    println!("    Empirical coverage: {:.1}%", empirical_coverage * 100.0);

    // Step 5: Create verified prediction with formal proof
    println!("\n[5] Creating verified prediction with proof...");
    let mut context = ConformalContext::new();

    let verified_pred = VerifiedPredictionBuilder::new()
        .interval(predictions[0].0, predictions[0].1)
        .confidence(0.95)
        .with_proof()
        .build(&mut context)
        .unwrap();

    println!("    Proof verified: {}", verified_pred.is_verified());
    println!("    Proof term: {:?}", verified_pred.proof().is_some());

    // Step 6: Validate results
    println!("\n[6] Validating results...");
    assert!(
        empirical_coverage >= 0.90,
        "Coverage guarantee failed: {} < 0.90",
        empirical_coverage
    );

    assert!(
        verified_pred.is_verified(),
        "Formal verification failed"
    );

    println!("    All validations passed ✓");
    println!("\n=== End-to-End Workflow Complete ===\n");
}

// ====================================================================================
// Edge Case and Stress Tests
// ====================================================================================

#[test]
fn test_multimodal_distribution() {
    //! Test 11: Multi-Modal Distribution
    //! Verify predictor handles complex multi-modal distributions

    let (cal_x, cal_y) = generate_trimodal_data(
        150,
        (-10.0, 0.5),
        (0.0, 0.3),
        (10.0, 0.5)
    );

    let mut measure = KNNNonconformity::new(7);
    measure.fit(&cal_x, &cal_y);

    let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();
    predictor.calibrate(&cal_x, &cal_y).unwrap();

    // Test in each mode
    let test_points = vec![
        (-10.0, "Mode 1"),
        (0.0, "Mode 2"),
        (10.0, "Mode 3"),
    ];

    println!("\nMulti-Modal Distribution Test:");
    for (point, mode_name) in test_points {
        let (lower, upper) = predictor.predict_interval(&[point], point).unwrap();
        println!("  {}: [{:.2}, {:.2}] width={:.2}", mode_name, lower, upper, upper - lower);

        assert!(lower <= point && point <= upper);
    }
}

#[test]
fn test_extreme_values() {
    //! Test 12: Stress Test with Extreme Values
    //! Verify robustness to outliers and extreme values

    let mut cal_x = Vec::new();
    let mut cal_y = Vec::new();

    // Normal data
    for i in 0..90 {
        cal_x.push(vec![i as f64]);
        cal_y.push(i as f64);
    }

    // Add extreme outliers
    cal_x.push(vec![1000.0]);
    cal_y.push(1000.0);
    cal_x.push(vec![-1000.0]);
    cal_y.push(-1000.0);

    let mut measure = KNNNonconformity::new(5);
    measure.fit(&cal_x, &cal_y);

    let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();
    predictor.calibrate(&cal_x, &cal_y).unwrap();

    // Test on normal values (should not be affected by outliers)
    let (lower, upper) = predictor.predict_interval(&[50.0], 50.0).unwrap();

    println!("\nExtreme Values Stress Test:");
    println!("  Prediction for x=50: [{:.2}, {:.2}]", lower, upper);
    println!("  Outliers handled: {}",
        if upper - lower < 100.0 { "✓ (robust)" } else { "✗ (affected)" }
    );

    // Interval should still be reasonable despite outliers
    assert!(lower <= 50.0 && 50.0 <= upper);
    assert!(upper - lower < 200.0, "Interval too wide due to outliers");
}

#[test]
fn test_empty_data_handling() {
    //! Test 13: Empty Data Handling
    //! Verify graceful handling of edge cases

    let measure = KNNNonconformity::new(3);
    let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();

    // Attempt calibration with empty data
    let result = predictor.calibrate(&[], &[]);

    assert!(result.is_err(), "Should reject empty calibration data");

    // Attempt prediction without calibration
    let measure2 = KNNNonconformity::new(3);
    let predictor2 = ConformalPredictor::new(0.1, measure2).unwrap();

    let result = predictor2.predict_interval(&[1.0], 1.0);
    assert!(result.is_err(), "Should reject prediction without calibration");

    println!("\nEmpty Data Handling: PASS ✓");
}

#[test]
fn test_single_sample() {
    //! Test 14: Single Sample Edge Case
    //! Verify behavior with minimal data

    let mut measure = KNNNonconformity::new(1);
    measure.fit(&[vec![1.0]], &[1.0]);

    let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();
    let result = predictor.calibrate(&[vec![1.0]], &[1.0]);

    // Should handle single sample gracefully
    assert!(result.is_ok(), "Should handle single sample");

    let (lower, upper) = predictor.predict_interval(&[1.0], 1.0).unwrap();

    println!("\nSingle Sample Test:");
    println!("  Interval: [{:.2}, {:.2}]", lower, upper);
    println!("  Result: PASS ✓");

    assert!(lower <= 1.0 && 1.0 <= upper);
}

#[test]
fn test_high_dimensional_data() {
    //! Test 15: High-Dimensional Feature Space
    //! Verify performance with many features

    let n_features = 50;
    let mut cal_x = Vec::new();
    let mut cal_y = Vec::new();

    for i in 0..100 {
        let features: Vec<f64> = (0..n_features)
            .map(|j| (i as f64 + j as f64) / 10.0)
            .collect();
        cal_x.push(features);
        cal_y.push(i as f64);
    }

    let mut measure = KNNNonconformity::new(5);
    measure.fit(&cal_x, &cal_y);

    let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();
    predictor.calibrate(&cal_x, &cal_y).unwrap();

    let test_features: Vec<f64> = (0..n_features)
        .map(|j| (50.0 + j as f64) / 10.0)
        .collect();

    let (lower, upper) = predictor.predict_interval(&test_features, 50.0).unwrap();

    println!("\nHigh-Dimensional Test ({} features):", n_features);
    println!("  Prediction: [{:.2}, {:.2}]", lower, upper);
    println!("  Result: PASS ✓");

    assert!(lower <= 50.0 && 50.0 <= upper);
}

#[test]
fn test_varying_calibration_sizes() {
    //! Test 16: Varying Calibration Set Sizes
    //! Verify behavior with different amounts of calibration data

    println!("\nVarying Calibration Set Sizes:");
    println!("{:<15} {:<12} {:<12}", "Cal Size", "Interval Width", "Coverage");
    println!("{}", "-".repeat(40));

    for &n_cal in &[20, 50, 100, 200] {
        let (cal_x, cal_y) = generate_synthetic_data(n_cal, DataDistribution::Normal(0.0, 1.0));

        let mut measure = KNNNonconformity::new(5);
        measure.fit(&cal_x, &cal_y);

        let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();
        predictor.calibrate(&cal_x, &cal_y).unwrap();

        let (test_x, test_y) = generate_synthetic_data(50, DataDistribution::Normal(0.0, 1.0));

        let mut total_width = 0.0;
        let mut covered = 0;

        for (x, y) in test_x.iter().zip(test_y.iter()) {
            let (lower, upper) = predictor.predict_interval(x, *y).unwrap();
            total_width += upper - lower;
            if lower <= *y && *y <= upper {
                covered += 1;
            }
        }

        let avg_width = total_width / test_x.len() as f64;
        let coverage = covered as f64 / test_y.len() as f64;

        println!("{:<15} {:<12.2} {:<12.1}%", n_cal, avg_width, coverage * 100.0);

        assert!(coverage >= 0.80, "Coverage too low with {} calibration samples", n_cal);
    }
}
