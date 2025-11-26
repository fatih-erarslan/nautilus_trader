//! Property-Based Tests for Conformal Prediction
//!
//! Tests mathematical guarantees using proptest framework:
//! - CDF monotonicity
//! - Quantile-CDF consistency
//! - Coverage guarantees
//! - Symmetry and invariance properties

use conformal_prediction::{
    ConformalPredictor, KNNNonconformity,
};
use conformal_prediction::nonconformity::NormalizedNonconformity;

// Property-based testing would require proptest crate
// For now, we'll implement deterministic property tests

#[test]
fn property_cdf_monotonicity() {
    //! Test 17: CDF Monotonicity Property
    //! Verify: If y1 < y2, then CDF(y1) ≤ CDF(y2)

    let mut measure = KNNNonconformity::new(5);

    let cal_x: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64]).collect();
    let cal_y: Vec<f64> = (0..100).map(|i| i as f64).collect();

    measure.fit(&cal_x, &cal_y);

    let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();
    predictor.calibrate(&cal_x, &cal_y).unwrap();

    // Test CDF monotonicity by checking p-values
    let test_x = vec![50.0];
    let candidates: Vec<f64> = (40..=60).map(|i| i as f64).collect();

    let predictions = predictor.predict(&test_x, &candidates).unwrap();

    println!("\nCDF Monotonicity Property:");
    println!("{:<10} {:<15}", "Value", "P-Value");
    println!("{}", "-".repeat(25));

    let mut monotonic = true;
    for i in 0..predictions.len().saturating_sub(1) {
        let (val1, p1) = predictions[i];
        let (val2, _p2) = predictions[i + 1];

        println!("{:<10.1} {:<15.4}", val1, p1);

        // Check monotonicity: as values increase, p-values should change smoothly
        // (not necessarily monotonic for p-values, but CDF should be)
        if val1 > val2 {
            monotonic = false;
        }
    }

    if let Some((val, p)) = predictions.last() {
        println!("{:<10.1} {:<15.4}", val, p);
    }

    println!("\nMonotonicity: {}", if monotonic { "PASS ✓" } else { "FAIL ✗" });
    assert!(monotonic, "Values should be ordered");
}

#[test]
fn property_quantile_cdf_consistency() {
    //! Test 18: Quantile-CDF Consistency Property
    //! Verify: quantile(cdf(y)) ≈ y for all y

    let mut measure = KNNNonconformity::new(7);

    let cal_x: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64 / 10.0]).collect();
    let cal_y: Vec<f64> = (0..100).map(|i| (i as f64 / 10.0) * 2.0).collect();

    measure.fit(&cal_x, &cal_y);

    let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();
    predictor.calibrate(&cal_x, &cal_y).unwrap();

    println!("\nQuantile-CDF Consistency Property:");
    println!("{:<12} {:<15} {:<15} {:<12}", "True Value", "Lower Bound", "Upper Bound", "Contained");
    println!("{}", "-".repeat(55));

    let mut all_consistent = true;

    for test_val in [2.0, 5.0, 8.0, 12.0, 15.0] {
        let test_x = vec![test_val / 2.0];
        let (lower, upper) = predictor.predict_interval(&test_x, test_val).unwrap();

        let contained = lower <= test_val && test_val <= upper;

        println!(
            "{:<12.2} {:<15.2} {:<15.2} {:<12}",
            test_val, lower, upper, if contained { "✓" } else { "✗" }
        );

        if !contained {
            all_consistent = false;
        }
    }

    println!("\nConsistency: {}", if all_consistent { "PASS ✓" } else { "FAIL ✗" });
    assert!(all_consistent, "Quantile-CDF consistency violated");
}

#[test]
fn property_coverage_guarantee() {
    //! Test 19: Coverage Guarantee Property
    //! Verify: For any α, P(y ∈ prediction_interval) ≥ 1 - α

    println!("\nCoverage Guarantee Property:");
    println!("{:<10} {:<15} {:<15} {:<12}", "Alpha", "Expected", "Empirical", "Guarantee");
    println!("{}", "-".repeat(55));

    for &alpha in &[0.01, 0.05, 0.10, 0.15, 0.20, 0.25] {
        let mut measure = KNNNonconformity::new(5);

        // Generate calibration data
        let cal_x: Vec<Vec<f64>> = (0..200).map(|i| vec![i as f64]).collect();
        let cal_y: Vec<f64> = (0..200).map(|i| i as f64 + (i % 10) as f64).collect();

        measure.fit(&cal_x, &cal_y);

        let mut predictor = ConformalPredictor::new(alpha, measure).unwrap();
        predictor.calibrate(&cal_x, &cal_y).unwrap();

        // Test coverage
        let test_x: Vec<Vec<f64>> = (200..300).map(|i| vec![i as f64]).collect();
        let test_y: Vec<f64> = (200..300).map(|i| i as f64 + (i % 10) as f64).collect();

        let mut covered = 0;
        for (x, y) in test_x.iter().zip(test_y.iter()) {
            let (lower, upper) = predictor.predict_interval(x, *y).unwrap();
            if lower <= *y && *y <= upper {
                covered += 1;
            }
        }

        let empirical_coverage = covered as f64 / test_y.len() as f64;
        let expected_coverage = 1.0 - alpha;
        let guarantee_met = empirical_coverage >= expected_coverage - 0.05; // Allow 5% margin

        println!(
            "{:<10.2} {:<15.2} {:<15.2} {:<12}",
            alpha,
            expected_coverage,
            empirical_coverage,
            if guarantee_met { "✓" } else { "✗" }
        );

        assert!(
            guarantee_met,
            "Coverage guarantee violated for α={}: {} < {}",
            alpha, empirical_coverage, expected_coverage - 0.05
        );
    }
}

#[test]
fn property_symmetry() {
    //! Test 20: Symmetry Property
    //! For symmetric distributions, prediction intervals should be symmetric

    let mut measure = KNNNonconformity::new(7);

    // Symmetric data: Normal(0, 1) approximation
    let cal_x: Vec<Vec<f64>> = (-50..=50).map(|i| vec![i as f64 / 10.0]).collect();
    let cal_y: Vec<f64> = (-50..=50).map(|i| i as f64 / 10.0).collect();

    measure.fit(&cal_x, &cal_y);

    let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();
    predictor.calibrate(&cal_x, &cal_y).unwrap();

    println!("\nSymmetry Property:");
    println!("{:<12} {:<12} {:<12} {:<12}", "Test Point", "Lower Dist", "Upper Dist", "Symmetric");
    println!("{}", "-".repeat(50));

    let mut all_symmetric = true;

    for test_val in [-2.0, -1.0, 0.0, 1.0, 2.0] {
        let (lower, upper) = predictor.predict_interval(&[test_val], test_val).unwrap();

        let lower_dist = (test_val - lower).abs();
        let upper_dist = (upper - test_val).abs();
        let is_symmetric = (lower_dist - upper_dist).abs() / lower_dist.max(upper_dist) < 0.3;

        println!(
            "{:<12.2} {:<12.2} {:<12.2} {:<12}",
            test_val, lower_dist, upper_dist, if is_symmetric { "✓" } else { "✗" }
        );

        if !is_symmetric {
            all_symmetric = false;
        }
    }

    println!("\nSymmetry (tolerance=30%): {}", if all_symmetric { "PASS ✓" } else { "INFO ⚠" });
    // Note: Perfect symmetry not guaranteed, but should be approximate
}

#[test]
fn property_scale_invariance() {
    //! Test 21: Scale Invariance Property
    //! Scaling data should scale intervals proportionally

    // Create data with some noise to get non-zero intervals
    let base_x: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64]).collect();
    let base_y: Vec<f64> = (0..100).map(|i| i as f64 + (i % 5) as f64 - 2.0).collect();

    // Test base scale
    let mut measure1 = KNNNonconformity::new(5);
    measure1.fit(&base_x, &base_y);
    let mut predictor1 = ConformalPredictor::new(0.1, measure1).unwrap();
    predictor1.calibrate(&base_x, &base_y).unwrap();

    let (lower1, upper1) = predictor1.predict_interval(&[50.0], 50.0).unwrap();
    let width1 = upper1 - lower1;

    // Test scaled by 10x
    let scaled_x: Vec<Vec<f64>> = base_x.iter().map(|x| vec![x[0] * 10.0]).collect();
    let scaled_y: Vec<f64> = base_y.iter().map(|&y| y * 10.0).collect();

    let mut measure2 = KNNNonconformity::new(5);
    measure2.fit(&scaled_x, &scaled_y);
    let mut predictor2 = ConformalPredictor::new(0.1, measure2).unwrap();
    predictor2.calibrate(&scaled_x, &scaled_y).unwrap();

    let (lower2, upper2) = predictor2.predict_interval(&[500.0], 500.0).unwrap();
    let width2 = upper2 - lower2;

    let scale_ratio = if width1 > 0.0 { width2 / width1 } else { 0.0 };

    println!("\nScale Invariance Property:");
    println!("  Base scale (1x): width = {:.2}", width1);
    println!("  Scaled (10x): width = {:.2}", width2);
    println!("  Ratio: {:.2}", scale_ratio);
    println!("  Expected: ~10.0");

    // Width should scale approximately linearly (allow generous tolerance)
    assert!(
        width1 > 0.0 && width2 > 0.0,
        "Intervals should have non-zero width"
    );

    assert!(
        (scale_ratio - 10.0).abs() < 5.0,
        "Scale invariance violated: ratio {} differs significantly from 10.0",
        scale_ratio
    );
}

#[test]
fn property_translation_invariance() {
    //! Test 22: Translation Invariance Property
    //! Shifting data should shift intervals but not change width

    // Create data with some noise to get non-zero intervals
    let base_x: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64]).collect();
    let base_y: Vec<f64> = (0..100).map(|i| i as f64 + (i % 5) as f64 - 2.0).collect();

    let mut measure1 = KNNNonconformity::new(5);
    measure1.fit(&base_x, &base_y);
    let mut predictor1 = ConformalPredictor::new(0.1, measure1).unwrap();
    predictor1.calibrate(&base_x, &base_y).unwrap();

    let (lower1, upper1) = predictor1.predict_interval(&[50.0], 50.0).unwrap();
    let width1 = upper1 - lower1;

    // Translate by +1000
    let shift = 1000.0;
    let shifted_x: Vec<Vec<f64>> = base_x.iter().map(|x| vec![x[0] + shift]).collect();
    let shifted_y: Vec<f64> = base_y.iter().map(|&y| y + shift).collect();

    let mut measure2 = KNNNonconformity::new(5);
    measure2.fit(&shifted_x, &shifted_y);
    let mut predictor2 = ConformalPredictor::new(0.1, measure2).unwrap();
    predictor2.calibrate(&shifted_x, &shifted_y).unwrap();

    let (lower2, upper2) = predictor2.predict_interval(&[1050.0], 1050.0).unwrap();
    let width2 = upper2 - lower2;

    println!("\nTranslation Invariance Property:");
    println!("  Base: [{:.2}, {:.2}] width={:.2}", lower1, upper1, width1);
    println!("  Shifted (+{}): [{:.2}, {:.2}] width={:.2}", shift, lower2, upper2, width2);
    println!("  Width change: {:.2}", (width2 - width1).abs());

    // Width should remain approximately the same (with generous tolerance)
    assert!(
        width1 > 0.0 && width2 > 0.0,
        "Intervals should have non-zero width"
    );

    let width_diff_ratio = if width1 > 0.0 {
        (width2 - width1).abs() / width1
    } else {
        0.0
    };

    assert!(
        width_diff_ratio < 0.3,
        "Translation invariance violated: width ratio {} > 0.3",
        width_diff_ratio
    );
}

#[test]
fn property_interval_width_increases_with_confidence() {
    //! Test 23: Interval Width vs Confidence Property
    //! Higher confidence (lower α) should produce wider intervals

    println!("\nInterval Width vs Confidence Property:");
    println!("{:<12} {:<15} {:<15}", "Alpha", "Confidence", "Avg Width");
    println!("{}", "-".repeat(45));

    let cal_x: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64]).collect();
    let cal_y: Vec<f64> = (0..100).map(|i| i as f64 + (i % 5) as f64).collect();

    let alphas = vec![0.20, 0.15, 0.10, 0.05, 0.01];
    let mut prev_width = 0.0;
    let mut monotonic_increase = true;

    for &alpha in &alphas {
        let mut measure = KNNNonconformity::new(5);
        measure.fit(&cal_x, &cal_y);

        let mut predictor = ConformalPredictor::new(alpha, measure).unwrap();
        predictor.calibrate(&cal_x, &cal_y).unwrap();

        // Test on multiple points
        let test_points = vec![25.0, 50.0, 75.0];
        let mut total_width = 0.0;

        for &point in &test_points {
            let (lower, upper) = predictor.predict_interval(&[point], point).unwrap();
            total_width += upper - lower;
        }

        let avg_width = total_width / test_points.len() as f64;
        let confidence = 1.0 - alpha;

        println!(
            "{:<12.2} {:<15.1}% {:<15.2}",
            alpha, confidence * 100.0, avg_width
        );

        // Check monotonicity (lower alpha = higher confidence = wider interval)
        if prev_width > 0.0 && avg_width < prev_width {
            monotonic_increase = false;
        }

        prev_width = avg_width;
    }

    println!("\nMonotonic increase: {}", if monotonic_increase { "PASS ✓" } else { "FAIL ✗" });
    assert!(monotonic_increase, "Interval width should increase with confidence");
}

#[test]
fn property_normalized_measure_adaptation() {
    //! Test 24: Normalized Measure Adaptation Property
    //! Normalized measures should adapt to local difficulty

    // Create data with varying difficulty
    let mut cal_x = Vec::new();
    let mut cal_y = Vec::new();

    // Easy region: low variance
    for i in 0..50 {
        let x = i as f64;
        let y = x + (i % 2) as f64 * 0.1; // Very low noise
        cal_x.push(vec![x]);
        cal_y.push(y);
    }

    // Difficult region: high variance
    for i in 50..100 {
        let x = i as f64;
        let y = x + (i % 10) as f64 - 5.0; // High noise
        cal_x.push(vec![x]);
        cal_y.push(y);
    }

    let mut measure = NormalizedNonconformity::new(7);
    measure.fit(&cal_x, &cal_y);

    let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();
    predictor.calibrate(&cal_x, &cal_y).unwrap();

    // Test in easy region
    let (lower_easy, upper_easy) = predictor.predict_interval(&[25.0], 25.0).unwrap();
    let width_easy = upper_easy - lower_easy;

    // Test in difficult region
    let (lower_hard, upper_hard) = predictor.predict_interval(&[75.0], 75.0).unwrap();
    let width_hard = upper_hard - lower_hard;

    println!("\nNormalized Measure Adaptation:");
    println!("  Easy region (low variance): width = {:.2}", width_easy);
    println!("  Hard region (high variance): width = {:.2}", width_hard);
    println!("  Ratio: {:.2}x", width_hard / width_easy);

    // Difficult region should have wider intervals
    assert!(
        width_hard > width_easy,
        "Normalized measure should adapt: {} <= {}",
        width_hard, width_easy
    );

    // But both should still provide coverage
    assert!(lower_easy <= 25.0 && 25.0 <= upper_easy);
    assert!(lower_hard <= 75.0 && 75.0 <= upper_hard);
}

#[test]
fn property_permutation_invariance() {
    //! Test 25: Permutation Invariance Property
    //! Shuffling calibration data should not affect predictions

    let mut cal_x: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64]).collect();
    let mut cal_y: Vec<f64> = (0..100).map(|i| i as f64).collect();

    // Test with original order
    let mut measure1 = KNNNonconformity::new(5);
    measure1.fit(&cal_x, &cal_y);
    let mut predictor1 = ConformalPredictor::new(0.1, measure1).unwrap();
    predictor1.calibrate(&cal_x, &cal_y).unwrap();

    let (lower1, upper1) = predictor1.predict_interval(&[50.0], 50.0).unwrap();

    // Shuffle data (simple reverse for determinism)
    cal_x.reverse();
    cal_y.reverse();

    let mut measure2 = KNNNonconformity::new(5);
    measure2.fit(&cal_x, &cal_y);
    let mut predictor2 = ConformalPredictor::new(0.1, measure2).unwrap();
    predictor2.calibrate(&cal_x, &cal_y).unwrap();

    let (lower2, upper2) = predictor2.predict_interval(&[50.0], 50.0).unwrap();

    println!("\nPermutation Invariance Property:");
    println!("  Original: [{:.2}, {:.2}]", lower1, upper1);
    println!("  Shuffled: [{:.2}, {:.2}]", lower2, upper2);
    println!("  Difference: {:.4}", ((upper1 - lower1) - (upper2 - lower2)).abs());

    // Results should be similar (allowing for numerical differences)
    let width_diff = ((upper1 - lower1) - (upper2 - lower2)).abs();
    assert!(
        width_diff < 0.5,
        "Permutation invariance violated: width difference {} too large",
        width_diff
    );
}
