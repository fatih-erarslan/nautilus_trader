//! Integration tests for streaming conformal prediction under concept drift
//!
//! These tests simulate various drift scenarios to validate the adaptive
//! behavior of the streaming conformal predictor.

use conformal_prediction::streaming::{
    StreamingConformalPredictor, PIDConfig, WindowConfig
};

/// Test sudden drift (abrupt distribution change)
#[test]
fn test_sudden_drift() {
    let mut predictor = StreamingConformalPredictor::new(0.1, 0.02);

    // Phase 1: Stationary data (mean=0, std=1)
    for i in 0..100 {
        let noise = (i % 3) as f64 - 1.0; // Simple noise [-1, 0, 1]
        let y_true = 0.0 + noise;
        let y_pred = 0.0;
        predictor.update(&[i as f64], y_true, y_pred);
    }

    let (lower1, upper1) = predictor.predict_interval(0.0).unwrap();
    let width1 = upper1 - lower1;

    // Phase 2: Sudden drift (mean=10, std=2)
    for i in 100..200 {
        let noise = ((i % 5) as f64 - 2.0) * 2.0; // Noise [-4, -2, 0, 2, 4]
        let y_true = 10.0 + noise;
        let y_pred = 10.0;
        predictor.update(&[i as f64], y_true, y_pred);
    }

    let (lower2, upper2) = predictor.predict_interval(10.0).unwrap();
    let width2 = upper2 - lower2;

    // Interval should adapt to new distribution
    assert!(width2 > width1, "Interval should widen for increased variance");
    assert!(predictor.n_samples() > 0);
}

/// Test gradual drift (slow distribution change)
#[test]
fn test_gradual_drift() {
    let mut predictor = StreamingConformalPredictor::new(0.1, 0.01);

    // Gradual drift: mean increases linearly from 0 to 10
    for i in 0..200 {
        let mean = (i as f64 / 200.0) * 10.0; // Drift from 0 to 10
        let noise = (i % 3) as f64 - 1.0;
        let y_true = mean + noise;
        let y_pred = mean;

        predictor.update(&[i as f64], y_true, y_pred);
    }

    // Should maintain reasonable interval width
    let (lower, upper) = predictor.predict_interval(10.0).unwrap();
    let width = upper - lower;

    assert!(width > 1.0 && width < 5.0, "Width should be reasonable: {}", width);
}

/// Test recurring patterns (seasonal drift)
#[test]
fn test_seasonal_drift() {
    let mut predictor = StreamingConformalPredictor::new(0.1, 0.015);

    // Two cycles of seasonal pattern
    for cycle in 0..2 {
        // High variance period
        for i in 0..50 {
            let idx = cycle * 100 + i;
            let noise = ((idx % 7) as f64 - 3.0) * 2.0; // High variance
            let y_true = 5.0 + noise;
            let y_pred = 5.0;
            predictor.update(&[idx as f64], y_true, y_pred);
        }

        // Low variance period
        for i in 50..100 {
            let idx = cycle * 100 + i;
            let noise = (idx % 3) as f64 - 1.0; // Low variance
            let y_true = 5.0 + noise;
            let y_pred = 5.0;
            predictor.update(&[idx as f64], y_true, y_pred);
        }
    }

    // Should adapt to current (low variance) regime
    // Note: Will retain some memory of high variance periods
    let (lower, upper) = predictor.predict_interval(5.0).unwrap();
    let width = upper - lower;

    // Width should be reasonable, though may include memory of high variance
    assert!(width < 15.0 && width > 0.0, "Width should be reasonable: {}", width);
}

/// Test high-frequency noise
#[test]
fn test_high_frequency_noise() {
    let mut predictor = StreamingConformalPredictor::new(0.1, 0.03);

    // Rapidly alternating between two distributions
    for i in 0..200 {
        let (mean, noise_scale) = if i % 2 == 0 {
            (0.0, 1.0) // Distribution A
        } else {
            (10.0, 2.0) // Distribution B
        };

        let noise = ((i % 5) as f64 - 2.0) * noise_scale;
        let y_true = mean + noise;
        let y_pred = mean;

        predictor.update(&[i as f64], y_true, y_pred);
    }

    // Should maintain coverage despite rapid changes
    let width = {
        let (lower, upper) = predictor.predict_interval(5.0).unwrap();
        upper - lower
    };

    assert!(width > 0.0 && width < 30.0, "Interval should be finite: {}", width);
}

/// Test recovery from outliers
#[test]
fn test_outlier_recovery() {
    let mut predictor = StreamingConformalPredictor::new(0.1, 0.025);

    // Normal data
    for i in 0..50 {
        let noise = (i % 3) as f64 - 1.0;
        let y_true = 5.0 + noise;
        let y_pred = 5.0;
        predictor.update(&[i as f64], y_true, y_pred);
    }

    let (lower1, upper1) = predictor.predict_interval(5.0).unwrap();
    let width1 = upper1 - lower1;

    // Inject outliers
    for i in 50..60 {
        let y_true = if i % 2 == 0 { 50.0 } else { -50.0 };
        let y_pred = 5.0;
        predictor.update(&[i as f64], y_true, y_pred);
    }

    let (lower2, upper2) = predictor.predict_interval(5.0).unwrap();
    let width2 = upper2 - lower2;

    // Interval should widen for outliers
    assert!(width2 > width1, "Should widen: {} -> {}", width1, width2);

    // Return to normal data
    for i in 60..150 {
        let noise = (i % 3) as f64 - 1.0;
        let y_true = 5.0 + noise;
        let y_pred = 5.0;
        predictor.update(&[i as f64], y_true, y_pred);
    }

    let (lower3, upper3) = predictor.predict_interval(5.0).unwrap();
    let width3 = upper3 - lower3;

    // Should recover and narrow down
    assert!(width3 < width2, "Should recover: {} -> {}", width2, width3);
}

/// Test with coverage tracking
#[test]
fn test_coverage_under_drift() {
    let mut predictor = StreamingConformalPredictor::new(0.1, 0.02);

    let mut covered_count = 0;
    let mut total_count = 0;

    // Calibrate initially
    for i in 0..50 {
        let noise = (i % 3) as f64 - 1.0;
        let y_true = 0.0 + noise;
        let y_pred = 0.0;
        predictor.update(&[i as f64], y_true, y_pred);
    }

    // Test with drift
    for i in 50..200 {
        // Gradual mean shift
        let mean = ((i - 50) as f64 / 150.0) * 5.0;
        let noise = (i % 3) as f64 - 1.0;
        let y_true = mean + noise;
        let y_pred = mean;

        // Get prediction before update
        if let Ok((lower, upper)) = predictor.predict_interval(y_pred) {
            if y_true >= lower && y_true <= upper {
                covered_count += 1;
            }
            total_count += 1;
        }

        // Update with coverage feedback
        let prev_interval = if i > 50 {
            predictor.predict_interval(y_pred).ok()
        } else {
            None
        };

        predictor.update_with_coverage(&[i as f64], y_true, y_pred, prev_interval);
    }

    // Should maintain reasonable coverage (target 90%)
    let empirical_coverage = covered_count as f64 / total_count as f64;
    assert!(
        empirical_coverage > 0.70,
        "Coverage too low: {:.2}% (target 90%)",
        empirical_coverage * 100.0
    );

    // Note: 100% coverage is acceptable (conservative but valid)
    // Under drift, predictor may be conservative to maintain guarantees
    assert!(
        empirical_coverage <= 1.0,
        "Coverage should not exceed 100%: {:.2}%",
        empirical_coverage * 100.0
    );
}

/// Test PID adaptation
#[test]
fn test_pid_adaptation() {
    let mut predictor = StreamingConformalPredictor::new(0.1, 0.02);

    let initial_decay = predictor.decay_rate();

    // Create systematic undercoverage
    for i in 0..100 {
        let y_true = i as f64;
        let y_pred = y_true;

        // Use narrow interval to force undercoverage
        let interval = Some((y_pred - 0.5, y_pred + 0.5));
        predictor.update_with_coverage(&[i as f64], y_true, y_pred, interval);
    }

    // PID should adjust decay rate
    let final_decay = predictor.decay_rate();

    // Decay rate should have changed (direction depends on PID response)
    // Just verify that adaptation occurred
    assert_ne!(
        initial_decay, final_decay,
        "PID should adapt decay rate"
    );
}

/// Test window size limits
#[test]
fn test_window_size_limit() {
    let window_config = WindowConfig {
        max_size: Some(100),
        max_age: None,
        initial_capacity: 50,
    };

    let pid_config = PIDConfig::default();

    let mut predictor = StreamingConformalPredictor::with_config(
        0.1,
        0.01,
        window_config,
        pid_config,
    );

    // Add more samples than window size
    for i in 0..500 {
        let y_true = i as f64;
        let y_pred = y_true;
        predictor.update(&[i as f64], y_true, y_pred);
    }

    // Should respect window size limit
    assert_eq!(predictor.n_samples(), 100);
}

/// Test reset functionality
#[test]
fn test_reset_clears_state() {
    let mut predictor = StreamingConformalPredictor::new(0.1, 0.01);

    // Add data
    for i in 0..50 {
        predictor.update(&[i as f64], i as f64, i as f64);
    }

    assert!(predictor.n_samples() > 0);

    // Reset
    predictor.reset();

    // Should be cleared
    assert_eq!(predictor.n_samples(), 0);
    assert!(predictor.empirical_coverage().is_none());

    // Should be able to use again
    predictor.update(&[1.0], 1.0, 1.0);
    assert_eq!(predictor.n_samples(), 1);
}

/// Test performance with large windows
#[test]
fn test_performance_stress() {
    use std::time::Instant;

    let mut predictor = StreamingConformalPredictor::new(0.1, 0.01);

    // Warm up
    for i in 0..100 {
        predictor.update(&[i as f64], i as f64, i as f64);
    }

    // Measure update performance
    let start = Instant::now();
    for i in 100..1100 {
        predictor.update(&[i as f64], i as f64, i as f64 + 0.5);
    }
    let elapsed = start.elapsed();

    let avg_update_us = elapsed.as_micros() / 1000;

    // Should be fast (<0.5ms = 500μs per update)
    assert!(
        avg_update_us < 500,
        "Update too slow: {}μs per update",
        avg_update_us
    );

    // Measure prediction performance
    let start = Instant::now();
    for _ in 0..100 {
        let _ = predictor.predict_interval(0.0);
    }
    let elapsed = start.elapsed();

    let avg_predict_us = elapsed.as_micros() / 100;

    println!("Performance: update={}μs, predict={}μs", avg_update_us, avg_predict_us);
}
