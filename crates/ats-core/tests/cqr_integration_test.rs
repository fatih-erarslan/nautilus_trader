//! Integration tests for CQR module
//!
//! These tests verify the complete CQR workflow and validate the mathematical
//! properties required by Romano et al. (2019).

use ats_core::cqr::{
    AsymmetricCqrCalibrator, AsymmetricCqrConfig, CqrCalibrator, CqrConfig, SymmetricCqr,
};

/// Test that CQR achieves target coverage on synthetic data
#[test]
fn test_cqr_coverage_guarantee() {
    let config = CqrConfig {
        alpha: 0.1, // 90% coverage
        symmetric: true,
    };

    let mut calibrator = CqrCalibrator::new(config);

    // Generate calibration data
    let n_cal = 100;
    let y_cal: Vec<f32> = (0..n_cal).map(|i| (i as f32) / 10.0).collect();
    let q_lo_cal: Vec<f32> = y_cal.iter().map(|y| y - 0.5).collect();
    let q_hi_cal: Vec<f32> = y_cal.iter().map(|y| y + 0.5).collect();

    calibrator.calibrate(&y_cal, &q_lo_cal, &q_hi_cal);

    // Test on separate test set
    let n_test = 50;
    let y_test: Vec<f32> = (0..n_test)
        .map(|i| (i as f32) / 10.0 + 10.0)
        .collect();
    let q_lo_test: Vec<f32> = y_test.iter().map(|y| y - 0.5).collect();
    let q_hi_test: Vec<f32> = y_test.iter().map(|y| y + 0.5).collect();

    let coverage = calibrator.compute_coverage(&y_test, &q_lo_test, &q_hi_test);

    // Coverage should be at least 1 - alpha = 0.9
    assert!(
        coverage >= 0.9,
        "Coverage {} is below target 0.9",
        coverage
    );

    println!("✅ CQR Coverage Test: {:.2}% (target ≥90%)", coverage * 100.0);
}

/// Test asymmetric CQR and compare with symmetric
#[test]
fn test_asymmetric_vs_symmetric() {
    // Calibration data
    let y_cal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let q_lo_cal: Vec<f32> = y_cal.iter().map(|y| y - 1.0).collect();
    let q_hi_cal: Vec<f32> = y_cal.iter().map(|y| y + 1.0).collect();

    // Symmetric CQR
    let mut symmetric = CqrCalibrator::new(CqrConfig {
        alpha: 0.1,
        symmetric: true,
    });
    symmetric.calibrate(&y_cal, &q_lo_cal, &q_hi_cal);

    // Asymmetric CQR
    let mut asymmetric = AsymmetricCqrCalibrator::new(AsymmetricCqrConfig {
        alpha: 0.1,
        alpha_lo: 0.05,
        alpha_hi: 0.05,
    });
    asymmetric.calibrate(&y_cal, &q_lo_cal, &q_hi_cal);

    // Compare intervals on test point
    let (sym_lo, sym_hi) = symmetric.predict_interval(5.0, 6.0);
    let (asym_lo, asym_hi) = asymmetric.predict_interval(5.0, 6.0);

    println!(
        "Symmetric interval: [{:.3}, {:.3}], width: {:.3}",
        sym_lo,
        sym_hi,
        sym_hi - sym_lo
    );
    println!(
        "Asymmetric interval: [{:.3}, {:.3}], width: {:.3}",
        asym_lo,
        asym_hi,
        asym_hi - asym_lo
    );

    // Both should produce valid intervals
    assert!(sym_lo < sym_hi);
    assert!(asym_lo < asym_hi);

    // Test coverage on same data
    let y_test = y_cal.clone();
    let q_lo_test = q_lo_cal.clone();
    let q_hi_test = q_hi_cal.clone();

    let sym_coverage = symmetric.compute_coverage(&y_test, &q_lo_test, &q_hi_test);
    let asym_coverage = asymmetric.compute_coverage(&y_test, &q_lo_test, &q_hi_test);

    println!("Symmetric coverage: {:.1}%", sym_coverage * 100.0);
    println!("Asymmetric coverage: {:.1}%", asym_coverage * 100.0);

    // Both should achieve target coverage
    assert!(sym_coverage >= 0.9);
    assert!(asym_coverage >= 0.9);
}

/// Test SymmetricCqr with diagnostic utilities
#[test]
fn test_symmetric_cqr_diagnostics() {
    let config = CqrConfig {
        alpha: 0.1,
        symmetric: true,
    };

    let mut cqr = SymmetricCqr::new(config);

    // Calibration
    let y_cal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let q_lo_cal = vec![0.5, 1.5, 2.5, 3.5, 4.5];
    let q_hi_cal = vec![1.5, 2.5, 3.5, 4.5, 5.5];

    cqr.calibrate(&y_cal, &q_lo_cal, &q_hi_cal);

    // Compute statistics
    let stats = cqr.compute_interval_statistics(&q_lo_cal, &q_hi_cal);

    println!("Interval Statistics:");
    println!("  Mean width: {:.3}", stats.mean_width);
    println!("  Median width: {:.3}", stats.median_width);
    println!("  Min width: {:.3}", stats.min_width);
    println!("  Max width: {:.3}", stats.max_width);
    println!("  Std dev: {:.3}", stats.std_width);

    assert!(stats.mean_width > 0.0);
    assert!(stats.min_width <= stats.median_width);
    assert!(stats.median_width <= stats.max_width);

    // Evaluate performance
    let metrics = cqr.evaluate(&y_cal, &q_lo_cal, &q_hi_cal);

    println!("\nEvaluation Metrics:");
    println!("  Coverage: {:.1}%", metrics.coverage * 100.0);
    println!("  Avg width: {:.3}", metrics.average_width);
    println!("  Efficiency: {:.3}", metrics.efficiency);

    assert!(metrics.coverage > 0.0);
    assert!(metrics.average_width > 0.0);
    assert!(metrics.efficiency > 0.0);
}

/// Test that CQR properly handles edge cases
#[test]
fn test_cqr_edge_cases() {
    let config = CqrConfig::default();
    let mut calibrator = CqrCalibrator::new(config);

    // Test with perfect quantile predictions (all scores should be 0)
    let y_cal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let q_lo_cal = vec![0.9, 1.9, 2.9, 3.9, 4.9];
    let q_hi_cal = vec![1.1, 2.1, 3.1, 4.1, 5.1];

    calibrator.calibrate(&y_cal, &q_lo_cal, &q_hi_cal);

    // All calibration scores should be ≤ 0 (y within intervals)
    let scores = calibrator.get_calibration_scores();
    for &score in scores {
        assert!(
            score <= 0.01, // Allow small floating point error
            "Score {} should be ≤ 0 for y within interval",
            score
        );
    }

    println!("✅ Edge case test passed: perfect predictions");
}

/// Test coverage with varying alpha levels
#[test]
fn test_varying_alpha_levels() {
    let alphas = vec![0.05, 0.10, 0.20];

    let y_cal: Vec<f32> = (0..100).map(|i| (i as f32) / 10.0).collect();
    let q_lo_cal: Vec<f32> = y_cal.iter().map(|y| y - 0.5).collect();
    let q_hi_cal: Vec<f32> = y_cal.iter().map(|y| y + 0.5).collect();

    let y_test: Vec<f32> = (0..50).map(|i| (i as f32) / 10.0 + 10.0).collect();
    let q_lo_test: Vec<f32> = y_test.iter().map(|y| y - 0.5).collect();
    let q_hi_test: Vec<f32> = y_test.iter().map(|y| y + 0.5).collect();

    println!("\nCoverage vs Alpha:");
    for &alpha in &alphas {
        let config = CqrConfig {
            alpha,
            symmetric: true,
        };
        let mut calibrator = CqrCalibrator::new(config);
        calibrator.calibrate(&y_cal, &q_lo_cal, &q_hi_cal);

        let coverage = calibrator.compute_coverage(&y_test, &q_lo_test, &q_hi_test);
        let target_coverage = 1.0 - alpha;

        println!(
            "  α={:.2}: coverage={:.1}%, target≥{:.1}%",
            alpha,
            coverage * 100.0,
            target_coverage * 100.0
        );

        assert!(
            coverage >= target_coverage - 0.05, // Allow 5% tolerance
            "Coverage {} below target {} for α={}",
            coverage,
            target_coverage,
            alpha
        );
    }
}

/// Performance benchmark for CQR
#[test]
fn test_cqr_performance() {
    use std::time::Instant;

    let config = CqrConfig::default();
    let mut calibrator = CqrCalibrator::new(config);

    // Large calibration set
    let n = 10000;
    let y_cal: Vec<f32> = (0..n).map(|i| (i as f32) / 100.0).collect();
    let q_lo_cal: Vec<f32> = y_cal.iter().map(|y| y - 1.0).collect();
    let q_hi_cal: Vec<f32> = y_cal.iter().map(|y| y + 1.0).collect();

    // Benchmark calibration
    let start = Instant::now();
    calibrator.calibrate(&y_cal, &q_lo_cal, &q_hi_cal);
    let calibration_time = start.elapsed();

    println!(
        "\nCalibration time for {} samples: {:.2}ms",
        n,
        calibration_time.as_secs_f64() * 1000.0
    );

    // Benchmark batch prediction
    let n_test = 1000;
    let q_lo_test: Vec<f32> = (0..n_test).map(|i| (i as f32) / 100.0).collect();
    let q_hi_test: Vec<f32> = q_lo_test.iter().map(|y| y + 2.0).collect();

    let start = Instant::now();
    let intervals = calibrator.predict_intervals_batch(&q_lo_test, &q_hi_test);
    let prediction_time = start.elapsed();

    println!(
        "Batch prediction time for {} samples: {:.2}ms",
        n_test,
        prediction_time.as_secs_f64() * 1000.0
    );
    println!(
        "Average time per prediction: {:.2}μs",
        (prediction_time.as_secs_f64() * 1_000_000.0) / n_test as f64
    );

    assert_eq!(intervals.len(), n_test);

    // Performance should be reasonable
    assert!(calibration_time.as_millis() < 1000); // <1s for 10k samples
    assert!(prediction_time.as_micros() < 10000); // <10ms for 1k predictions
}
